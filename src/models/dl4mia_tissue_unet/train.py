import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm
from src.models.dl4mia_tissue_unet.my_utils import print_dict
from src.models.dl4mia_tissue_unet import config as cfg
from src.models.dl4mia_tissue_unet.dataset import TwoDimensionalDataset
from src.models.dl4mia_tissue_unet.model import UNet
from src.models.dl4mia_tissue_unet.utils import AverageMeter, Logger, Visualizer
from src.models.dl4mia_tissue_unet.utils2 import matching_dataset
from scipy import ndimage
torch.backends.cudnn.benchmark = True


def calculate_IoU(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection / union
        return iou


def save_checkpoint(state, is_best, save_dir, name='checkpoint.pth'):
    print('=> saving checkpoint', flush=True)
    file_name = os.path.join(save_dir, name)
    torch.save(state, file_name)

    if is_best:
        shutil.copyfile(file_name, os.path.join(
            save_dir, 'best_iou_model.pth'))

class Trainer():

    def __init__(self,train_dataset_dict, val_dataset_dict, model_dict, configs, color_map='magma'):
        self.configs = configs
        device = torch.device("cuda:0" if self.configs['cuda'] else "cpu")
        
        print('loading datasets...')
        self.train_dataset_it = self._create_data_loader(train_dataset_dict, shuffle=True, drop_last=True)
        self.val_dataset_it = self._create_data_loader(val_dataset_dict, shuffle=True, drop_last=True)

        print('\ncreating model...')
        self.model = self._create_model(model_dict, device)
        self.criterion = self._create_criterion(device, weight=[1.0, 1.0])
        self.optimizer = self._set_optimizer(weight_decay=1e-4)
        
        self.color_map = color_map

    def _create_data_loader(self, dataset_dict, shuffle=True, drop_last=True):
        dataset = TwoDimensionalDataset(**dataset_dict['kwargs'])
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_dict['batch_size'],
                                                    shuffle=shuffle, drop_last=drop_last,
                                                    num_workers=dataset_dict['workers'],
                                                    pin_memory=True if self.configs['cuda'] else False)
        return data_loader

    def _create_model(self, model_dict, device):
        # set model
        model = UNet(**model_dict['kwargs'])
        model = torch.nn.DataParallel(model).to(device)
        return model

    def _create_criterion(self, device:torch.device, weight:list=[1.0, 1.0, 5.0]):
        if self.configs['cuda']:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight).cuda())
        else:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight))
        criterion = nn.BCEWithLogitsLoss()
        criterion = torch.nn.DataParallel(criterion).to(device)
        return criterion

    def _set_optimizer(self, weight_decay:float=1e-4):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.configs['train_lr'], weight_decay=weight_decay)
        return optimizer

    def begin_training(self):
        
        if self.configs['save']:
            if not os.path.exists(self.configs['save_dir']):
                os.makedirs(self.configs['save_dir'])

        def lambda_(epoch):
            return pow((1 - ((epoch) / 200)), 0.9)

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_, )

        # Logger
        logger = Logger(('train', 'val', 'ap'), 'loss')

        # resume
        start_epoch = 0
        best_ap = 0
        if self.configs['resume_path'] is not None and os.path.exists(self.configs['resume_path']):
            print('Resuming model from {}'.format(self.configs['resume_path']), flush=True)
            state = torch.load(self.configs['resume_path'])
            start_epoch = state['epoch'] + 1
            best_ap = state['best_ap']
            self.model.load_state_dict(state['model_state_dict'], strict=True)
            self.optimizer.load_state_dict(state['optim_state_dict'])
            logger.data = state['logger_data']

        for epoch in range(start_epoch, start_epoch + self.configs['n_epochs']):
            print('Starting epoch {}'.format(epoch), flush=True)
            train_loss = self.train()
            val_loss, val_ap = self.val()

            scheduler.step()
            print('===> train loss: {:.2f}'.format(train_loss), flush=True)
            print('===> val loss: {:.2f}, val ap: {:.2f}'.format(val_loss, val_ap), flush=True)

            logger.add('train', train_loss)
            logger.add('val', val_loss)
            logger.add('ap', val_ap)
            logger.plot(save=self.configs['save'], save_dir=self.configs['save_dir'])

            is_best = val_ap > best_ap
            best_ap = max(val_ap, best_ap)

            if self.configs['save']:
                state = {
                    'epoch': epoch,
                    'best_ap': best_ap,
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optimizer.state_dict(),
                    'logger_data': logger.data,
                }
            save_checkpoint(state, is_best, save_dir=self.configs['save_dir'])

    def train(self):
        loss_meter = AverageMeter()
        self.model.train()
        for param_group in self.optimizer.param_groups:
            print('learning rate: {}'.format(param_group['lr']), flush=True)
        for i, sample in enumerate(tqdm(self.train_dataset_it)):
            images = sample['image']  # B 1 Z Y X
            semantic_masks = sample['semantic_mask']  # B 1 Z Y X
            # semantic_masks.squeeze_(1)  #FIXME B Z Y X (loss expects this format) JO But it doesn't match output
            output = self.model(images)  # B 3 Z Y X
            # loss = self.criterion(output, semantic_masks.long())  # B 1 Z Y X
            loss = self.criterion(output, semantic_masks.float())  # B 1 Z Y X JO only for  BCEWithLogitsLoss
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_meter.update(loss.item())

        return loss_meter.avg

    def val(self):
        loss_meter = AverageMeter()
        average_precision_meter = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(tqdm(self.val_dataset_it)):

                images = sample['image']  # B 1 Z Y X
                semantic_masks = sample['semantic_mask']  # B 1 Z Y X
                # semantic_masks.squeeze_(1)  #FIXME B Z Y X (loss expects this format) JO But it doesn't match output
                output = self.model(images)  # B 3 Z Y X
                # loss = self.criterion(output, semantic_masks.long())  # B 1 Z Y X
                loss = self.criterion(output, semantic_masks.float())  # B 1 Z Y X JO only for  BCEWithLogitsLoss
                loss = loss.mean()
                loss_meter.update(loss.item())
                # for b in range(output.shape[0]):
                #     output_softmax = F.softmax(output[b], dim=0)
                #     prediction_fg = output_softmax[0, ...].cpu().detach().numpy()
                #     # prediction_fg = output_softmax[1, ...].cpu().detach().numpy()
                #     pred_fg_thresholded = prediction_fg > 0.5
                #     sc = matching_dataset(y_pred=[pred_fg_thresholded], y_true=[semantic_masks[b, 0, ...].cpu().detach().numpy()])
                #     # instance_map, _ = ndimage.label(pred_fg_thresholded)
                #     # sc = matching_dataset(y_pred=[instance_map], y_true=[instance[b, 0, ...].cpu().detach().numpy()],
                #     #                     thresh=0.5, show_progress=False)
                #     average_precision_meter.update(sc.accuracy)

        return loss_meter.avg, average_precision_meter.avg

def main():
    train_dataset_dict = cfg.TRAIN_DATASET_DICT
    print('Training dictionary:')
    print_dict(train_dataset_dict)
    val_dataset_dict = cfg.VAL_DATASET_DICT
    print("Validation dictionary:")
    print_dict(val_dataset_dict)
    model_dict = cfg.MODEL_DICT
    print("Model dictionary:")
    print_dict(model_dict)
    configs = cfg.CONFIG_DICT
    print("Config dictionary:")
    print_dict(configs)
    model_trainer = Trainer(train_dataset_dict, val_dataset_dict, model_dict, configs)
    model_trainer.begin_training()

if __name__ == '__main__':
    main()

# import os
# import shutil
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from matplotlib import pyplot as plt
# from tqdm import tqdm

# from PixelClassification.datasets import get_dataset
# from PixelClassification.models import get_model
# from PixelClassification.utils.utils import AverageMeter, Logger, Visualizer
# from PixelClassification.utils2 import matching_dataset
# from scipy import ndimage
# torch.backends.cudnn.benchmark = True

# def train():
#     loss_meter = AverageMeter()
#     model.train()
#     for param_group in optimizer.param_groups:
#         print('learning rate: {}'.format(param_group['lr']), flush=True)
#     for i, sample in enumerate(tqdm(train_dataset_it)):
#         images = sample['image']  # B 1 Z Y X
#         semantic_masks = sample['semantic_mask']  # B 1 Z Y X
#         semantic_masks.squeeze_(1)  # B Z Y X (loss expects this format)
#         instance = sample['instance_mask']
#         output = model(images)  # B 3 Z Y X
#         loss = criterion(output, semantic_masks.long())  # B 1 Z Y X
#         loss = loss.mean()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         loss_meter.update(loss.item())

#     return loss_meter.avg





# def val():
#     loss_meter = AverageMeter()
#     average_precision_meter = AverageMeter()
#     model.eval()
#     with torch.no_grad():
#         for i, sample in enumerate(tqdm(val_dataset_it)):

#             images = sample['image']  # B 1 Z Y X
#             semantic_masks = sample['semantic_mask']  # B 1 Z Y X ==> channel can be 0, 1, 2
#             instance = sample['instance_mask']
#             semantic_masks.squeeze_(1)  # B Z Y X (loss expects this format)

#             output = model(images)  # B 3 Z Y X
#             loss = criterion(output, semantic_masks.long())  # B 1 Z Y X
#             loss = loss.mean()
#             loss_meter.update(loss.item())
#             for b in range(output.shape[0]):
#                 output_softmax = F.softmax(output[b], dim=0)
#                 prediction_fg = output_softmax[1, ...].cpu().detach().numpy()
#                 pred_fg_thresholded = prediction_fg > 0.5
#                 instance_map, _ = ndimage.label(pred_fg_thresholded)
#                 sc = matching_dataset(y_pred=[instance_map], y_true=[instance[b, 0, ...].cpu().detach().numpy()],
#                                       thresh=0.5, show_progress=False)
#                 average_precision_meter.update(sc.accuracy)

#     return loss_meter.avg, average_precision_meter.avg


# def calculate_IoU(pred, label):
#     intersection = ((label == 1) & (pred == 1)).sum()
#     union = ((label == 1) | (pred == 1)).sum()
#     if not union:
#         return 0
#     else:
#         iou = intersection / union
#         return iou


# def save_checkpoint(state, is_best, save_dir, name='checkpoint.pth'):
#     print('=> saving checkpoint', flush=True)
#     file_name = os.path.join(save_dir, name)
#     torch.save(state, file_name)

#     if is_best:
#         shutil.copyfile(file_name, os.path.join(
#             save_dir, 'best_iou_model.pth'))


# def begin_training(train_dataset_dict, val_dataset_dict, model_dict, configs, color_map='magma'):
    
    

#     if configs['save']:
#         if not os.path.exists(configs['save_dir']):
#             os.makedirs(configs['save_dir'])

#     # set device
#     device = torch.device("cuda:0" if configs['cuda'] else "cpu")

#     # define global variables
#     global train_dataset_it, val_dataset_it, model, criterion, optimizer, visualizer

#     # train dataloader

#     train_dataset = get_dataset(train_dataset_dict['name'], train_dataset_dict['kwargs'])
#     train_dataset_it = torch.utils.data.DataLoader(train_dataset, batch_size=train_dataset_dict['batch_size'],
#                                                    shuffle=True, drop_last=True,
#                                                    num_workers=train_dataset_dict['workers'],
#                                                    pin_memory=True if configs['cuda'] else False)

#     # val dataloader
#     val_dataset = get_dataset(val_dataset_dict['name'], val_dataset_dict['kwargs'])
#     val_dataset_it = torch.utils.data.DataLoader(val_dataset, batch_size=val_dataset_dict['batch_size'], shuffle=True,
#                                                  drop_last=True, num_workers=val_dataset_dict['workers'],
#                                                  pin_memory=True if configs['cuda'] else False)

#     # set model
#     model = get_model(model_dict['name'], model_dict['kwargs'])
#     model = torch.nn.DataParallel(model).to(device)

#     # create criterion
#     criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 5.0]).cuda())
#     criterion = torch.nn.DataParallel(criterion).to(device)

#     # set optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=configs['train_lr'], weight_decay=1e-4)

#     def lambda_(epoch):
#         return pow((1 - ((epoch) / 200)), 0.9)

#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_, )

#     # Logger
#     logger = Logger(('train', 'val', 'ap'), 'loss')

#     # resume
#     start_epoch = 0
#     best_ap = 0
#     if configs['resume_path'] is not None and os.path.exists(configs['resume_path']):
#         print('Resuming model from {}'.format(configs['resume_path']), flush=True)
#         state = torch.load(configs['resume_path'])
#         start_epoch = state['epoch'] + 1
#         best_ap = state['best_ap']
#         model.load_state_dict(state['model_state_dict'], strict=True)
#         optimizer.load_state_dict(state['optim_state_dict'])
#         logger.data = state['logger_data']

#     for epoch in range(start_epoch, start_epoch + configs['n_epochs']):
#         print('Starting epoch {}'.format(epoch), flush=True)
#         train_loss = train()
#         val_loss, val_ap = val()

#         scheduler.step()
#         print('===> train loss: {:.2f}'.format(train_loss), flush=True)
#         print('===> val loss: {:.2f}, val ap: {:.2f}'.format(val_loss, val_ap), flush=True)

#         logger.add('train', train_loss)
#         logger.add('val', val_loss)
#         logger.add('ap', val_ap)
#         logger.plot(save=configs['save'], save_dir=configs['save_dir'])

#         is_best = val_ap > best_ap
#         best_ap = max(val_ap, best_ap)

#         if configs['save']:
#             state = {
#                 'epoch': epoch,
#                 'best_ap': best_ap,
#                 'model_state_dict': model.state_dict(),
#                 'optim_state_dict': optimizer.state_dict(),
#                 'logger_data': logger.data,
#             }
#         save_checkpoint(state, is_best, save_dir=configs['save_dir'])
