import os
import torch
import torch.nn as nn
from tqdm import tqdm
from src.models.dl4mia_tissue_unet.config import Config
from src.models.dl4mia_tissue_unet.dataset import TwoDimensionalDataset
from src.models.dl4mia_tissue_unet.model import UNet
from src.models.dl4mia_tissue_unet.dl4mia_utils.general import print_dict, save_yaml
from src.models.dl4mia_tissue_unet.dl4mia_utils.metrics import binary_sem_seg_metrics
from src.models.dl4mia_tissue_unet.dl4mia_utils.train_utils import save_checkpoint, AverageMeter, Logger
torch.backends.cudnn.benchmark = True


class Trainer():

    def __init__(self,train_dataset_dict, val_dataset_dict, test_dataset_dict, model_dict, config_dict):
        """
        Args
            train_dataset_dict: configuration dictionary for training dataset
            val_dataset_dict: configuration dictionary for validation dataset
            test_dataset_dict: configuration dictionary for test dataset
            model_dict: configuration dictionary for model creation
            config_dict: configuration dictionary for training
        """
        self.train_dataset_dict = train_dataset_dict
        self.val_dataset_dict = val_dataset_dict
        self.test_dataset_dict = test_dataset_dict
        self.model_dict = model_dict
        self.config_dict = config_dict
        self.device = torch.device("cuda:0" if self.config_dict['cuda'] else "cpu")
        
        print('loading datasets...')
        self.train_dataset_it = self._create_data_loader(self.train_dataset_dict, shuffle=True, drop_last=True)
        self.val_dataset_it = self._create_data_loader(self.val_dataset_dict, shuffle=True, drop_last=True)

        print('\ncreating model...')
        self.model = self._create_model(model_dict)
        self.criterion = self._create_BCE_criterion()
        self.optimizer = self._set_optimizer(weight_decay=1e-4)
        

    def _create_data_loader(self, dataset_dict, shuffle=True, drop_last=True):
        dataset = TwoDimensionalDataset(**dataset_dict['kwargs'])
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_dict['batch_size'],
                                                    shuffle=shuffle, drop_last=drop_last,
                                                    num_workers=dataset_dict['workers'],
                                                    pin_memory=True if self.config_dict['cuda'] else False)
        return data_loader

    def _create_model(self, model_dict):
        # set model
        model = UNet(**model_dict['kwargs'])
        model = torch.nn.DataParallel(model)
        model.to(self.device)
        return model

    def _create_CE_criterion(self, weight:list=[1.0, 1.0, 5.0]):
        if self.config_dict['cuda']:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight).cuda())
        else:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight))
        criterion = torch.nn.DataParallel(criterion)
        criterion.to(self.device)
        return criterion

    def _create_BCE_criterion(self):
        criterion = nn.BCEWithLogitsLoss()
        criterion = torch.nn.DataParallel(criterion)
        criterion.to(self.device)
        return criterion

    def _set_optimizer(self, weight_decay:float=1e-4):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config_dict['train_lr'], weight_decay=weight_decay)
        return optimizer

    def begin_training(self):
        
        if self.config_dict['save']:
            if not os.path.exists(self.config_dict['save_dir']):
                os.makedirs(self.config_dict['save_dir'])
            save_yaml(dict_=self.train_dataset_dict, filepath=f"{self.config_dict['save_dir']}/train_dataset_dict.yaml")
            save_yaml(dict_=self.val_dataset_dict, filepath=f"{self.config_dict['save_dir']}/val_dataset_dict.yaml")
            save_yaml(dict_=self.test_dataset_dict, filepath=f"{self.config_dict['save_dir']}/test_dataset_dict.yaml")
            save_yaml(dict_=self.model_dict, filepath=f"{self.config_dict['save_dir']}/model_dict.yaml")
            save_yaml(dict_=self.config_dict, filepath=f"{self.config_dict['save_dir']}/config_dict.yaml")


        def lambda_(epoch):
            return pow((1 - ((epoch) / 200)), 0.9)

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_, )

        # Logger
        logger = Logger(('train', 'val', 'ap'), 'loss')

        # resume
        start_epoch = 0
        best_ap = 0
        if self.config_dict['resume_path'] is not None and os.path.exists(self.config_dict['resume_path']):
            print('Resuming model from {}'.format(self.config_dict['resume_path']), flush=True)
            state = torch.load(self.config_dict['resume_path'])
            start_epoch = state['epoch'] + 1
            best_ap = state['best_ap']
            self.model.load_state_dict(state['model_state_dict'], strict=True)
            self.optimizer.load_state_dict(state['optim_state_dict'])
            logger.data = state['logger_data']

        for epoch in range(start_epoch, start_epoch + self.config_dict['n_epochs']):
            print('Starting epoch {}'.format(epoch), flush=True)
            train_loss = self.train()
            val_loss, val_ap = self.val()

            scheduler.step()
            print('===> train loss: {:.2f}'.format(train_loss), flush=True)
            print('===> val loss: {:.2f}, val ap: {:.2f}'.format(val_loss, val_ap), flush=True)

            logger.add('train', train_loss)
            logger.add('val', val_loss)
            logger.add('ap', val_ap)
            logger.plot(save=self.config_dict['save'], save_dir=self.config_dict['save_dir'])

            is_best = val_ap > best_ap
            best_ap = max(val_ap, best_ap)

            if self.config_dict['save']:
                trainable_state = {
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_ap': val_ap,
                    'best_ap': best_ap,
                    'train_cuda': self.config_dict['cuda'],
                    'model_dict': self.model_dict,
                    'model_state_dict': self.model.module.state_dict(), # remove .module if not using nn.DataParallel model
                    'optim_state_dict': self.optimizer.state_dict(),
                    'logger_data': logger.data,
                }
                save_checkpoint(trainable_state, is_best, save_dir=self.config_dict['save_dir'], name='last.pth')

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
                # bin_metric = BinaryMetrics(activation='0-1') # Does a sigmoid and threshold activation
                # acc, dice, prec, spec, rec = bin_metric(semantic_masks[:, 0, ...], output)
                # average_precision_meter.update(dice)
                for b in range(output.shape[0]):
                    output_softmax = torch.sigmoid(output[b])
                    prediction_fg = output_softmax[0, ...].cpu().detach().numpy()
                    pred_fg_thresholded = (prediction_fg > 0.5).astype(int)
                    acc, dice, prec, spec, rec = binary_sem_seg_metrics(y_true=semantic_masks[b, 0, ...].cpu().detach().numpy(), y_pred=pred_fg_thresholded)
                    # sc = matching_dataset(y_pred=[pred_fg_thresholded], y_true=[semantic_masks[b, 0, ...].cpu().detach().numpy()])
                    # instance_map, _ = ndimage.label(pred_fg_thresholded)
                #     # sc = matching_dataset(y_pred=[instance_map], y_true=[instance[b, 0, ...].cpu().detach().numpy()],
                #     #                     thresh=0.5, show_progress=False)
                    average_precision_meter.update(dice)

        return loss_meter.avg, average_precision_meter.avg

def main(
    data_dir: str = "data/processed/uncropped",
    output_dir: str = "src/models/dl4mia_tissue_unet/results",
    n_channels: int = 1,
    n_classes: int = 1,
    n_levels: int = 3,
    in_size: tuple = (128, 128),
    init_lr: float = 5e-4,
    n_epochs: int = 4,
    batch_size: int = 16,
    save: bool = True,
):
    cfg = Config(
        data_dir=data_dir,
        output_dir=output_dir,
        n_channels=n_channels,
        n_classes=n_classes,
        n_levels=n_levels,
        in_size=in_size,
        init_lr=init_lr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        save=save
        )
    train_dataset_dict = cfg.TRAIN_DATASET_DICT
    print('Training dictionary:')
    print_dict(train_dataset_dict)
    val_dataset_dict = cfg.VAL_DATASET_DICT
    print("\nValidation dictionary:")
    print_dict(val_dataset_dict)
    test_dataset_dict = cfg.TEST_DATASET_DICT
    print("\nTest dictionary:")
    print_dict(test_dataset_dict)
    model_dict = cfg.MODEL_DICT
    print("\nModel dictionary:")
    print_dict(model_dict)
    config_dict = cfg.CONFIG_DICT
    print("\nConfig dictionary:")
    print_dict(config_dict)
    print()
    model_trainer = Trainer(train_dataset_dict, val_dataset_dict, test_dataset_dict, model_dict, config_dict)
    model_trainer.begin_training()

if __name__ == '__main__':
    main()
