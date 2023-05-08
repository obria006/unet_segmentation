import os
import torch
import torch.nn as nn
from tqdm import tqdm
from src.models.dl4mia_tissue_unet.config import Config
from src.models.dl4mia_tissue_unet.dataset import TwoDimensionalDataset
from src.models.dl4mia_tissue_unet.model_v2 import UNet
from src.models.dl4mia_tissue_unet.dl4mia_utils.general import print_dict, save_yaml
from src.models.dl4mia_tissue_unet.dl4mia_utils.metrics import binary_sem_seg_metrics, BinaryMetrics, SegmentationMetrics
from src.models.dl4mia_tissue_unet.dl4mia_utils.train_utils import (
    save_checkpoint,
    AverageMeter,
    Logger,
)
import src.models.dl4mia_tissue_unet.dl4mia_utils.loss as loss_utils

torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(
        self,
        train_dataset_dict,
        val_dataset_dict,
        test_dataset_dict,
        model_dict,
        config_dict,
    ):
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
        self.device = torch.device("cuda:0" if self.config_dict["cuda"] else "cpu")

        print("loading datasets...")
        self.train_dataset_it = self._create_data_loader(
            self.train_dataset_dict, shuffle=True, drop_last=True
        )
        self.val_dataset_it = self._create_data_loader(
            self.val_dataset_dict, shuffle=True, drop_last=True
        )

        print("\ncreating model...")
        self.model = self._create_model(model_dict)
        self.criterion = self._create_tversky_criterion()
        self.optimizer = self._set_optimizer(weight_decay=1e-4)

    def _create_data_loader(self, dataset_dict, shuffle=True, drop_last=True):
        dataset = TwoDimensionalDataset(**dataset_dict["kwargs"])
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_dict["batch_size"],
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=dataset_dict["workers"],
            pin_memory=True if self.config_dict["cuda"] else False,
        )
        return data_loader

    def _create_model(self, model_dict):
        # set model
        model = UNet(**model_dict["kwargs"])
        # model = torch.nn.DataParallel(model)
        model.to(self.device)
        return model

    def _create_CE_criterion(self, weight: list = [1.0, 1.0, 5.0]):
        if self.config_dict["cuda"]:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight).cuda())
        else:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight))
        # criterion = torch.nn.DataParallel(criterion)
        criterion.to(self.device)
        return criterion

    def _create_BCE_criterion(self):
        criterion = nn.BCEWithLogitsLoss()
        # criterion = torch.nn.DataParallel(criterion)
        criterion.to(self.device)
        return criterion
    
    def _create_tversky_criterion(self):
        if self.model_dict["kwargs"]["num_classes"] > 1:
            mode = "multiclass"
        else:
            mode = "binary"
        criterion = loss_utils.TverskyLoss(mode=mode, from_logits=True)
        # criterion = torch.nn.DataParallel(criterion)
        criterion.to(self.device)
        return criterion


    def _set_optimizer(self, weight_decay: float = 1e-4):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config_dict["train_lr"],
            weight_decay=weight_decay,
        )
        return optimizer

    def begin_training(self):
        if self.config_dict["save"]:
            if not os.path.exists(self.config_dict["save_dir"]):
                os.makedirs(self.config_dict["save_dir"])
            save_yaml(
                dict_=self.train_dataset_dict,
                filepath=f"{self.config_dict['save_dir']}/train_dataset_dict.yaml",
            )
            save_yaml(
                dict_=self.val_dataset_dict,
                filepath=f"{self.config_dict['save_dir']}/val_dataset_dict.yaml",
            )
            save_yaml(
                dict_=self.test_dataset_dict,
                filepath=f"{self.config_dict['save_dir']}/test_dataset_dict.yaml",
            )
            save_yaml(
                dict_=self.model_dict,
                filepath=f"{self.config_dict['save_dir']}/model_dict.yaml",
            )
            save_yaml(
                dict_=self.config_dict,
                filepath=f"{self.config_dict['save_dir']}/config_dict.yaml",
            )

        def lambda_(epoch):
            return pow((1 - ((epoch) / 200)), 0.9)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda_,
        )

        # Logger
        logger = Logger(("train", "val", "ap", "dice"), "loss")

        # resume
        start_epoch = 0
        best_dice = 0
        best_loss = 0
        if self.config_dict["resume_path"] is not None and os.path.exists(
            self.config_dict["resume_path"]
        ):
            print(
                "Resuming model from {}".format(self.config_dict["resume_path"]),
                flush=True,
            )
            state = torch.load(self.config_dict["resume_path"])
            start_epoch = state["epoch"] + 1
            best_dice = state["best_dice"]
            best_loss = state["best_loss"]
            self.model.load_state_dict(state["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(state["optim_state_dict"])
            logger.data = state["logger_data"]

        for epoch in range(start_epoch, start_epoch + self.config_dict["n_epochs"]):
            print("Starting epoch {}".format(epoch), flush=True)
            train_loss = self.train()
            val_loss, val_ap, val_dice = self.val()
            scheduler.step()
            print("===> train loss:\t{:.2f}".format(train_loss), flush=True)
            print("===> val loss:\t\t{:.2f}".format(val_loss), flush=True)
            print("===> val ap:\t\t{:.2f}".format(val_ap), flush=True)
            print("===> val dice:\t\t{:.2f}".format(val_dice), flush=True)

            logger.add("train", train_loss)
            logger.add("val", val_loss)
            logger.add("ap", val_ap)
            logger.add('dice', val_dice)
            logger.plot(
                save=self.config_dict["save"], save_dir=self.config_dict["save_dir"]
            )

            is_best = val_ap > best_dice
            best_dice = max(val_dice, best_dice)
            best_loss = min(val_loss, best_loss)

            if self.config_dict["save"]:
                if isinstance(self.model, torch.nn.DataParallel):
                    # remove .module if not using nn.DataParallel model
                    model_state_dict = self.model.module.state_dict()
                else:
                    model_state_dict = self.model.state_dict()
                trainable_state = {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_ap": val_ap,
                    "val_dice": val_dice,
                    "best_loss": best_loss,
                    "best_dice": best_dice,
                    "train_cuda": self.config_dict["cuda"],
                    "model_dict": self.model_dict,
                    "model_state_dict": model_state_dict,
                    "optim_state_dict": self.optimizer.state_dict(),
                    "logger_data": logger.data,
                }
                save_checkpoint(
                    trainable_state,
                    is_best,
                    save_dir=self.config_dict["save_dir"],
                    name="last.pth",
                )

    def train(self):
        loss_meter = AverageMeter()
        self.model.train()
        for param_group in self.optimizer.param_groups:
            print("learning rate: {}".format(param_group["lr"]), flush=True)
        for i, sample in enumerate(tqdm(self.train_dataset_it)):
            images = sample["image"].to(self.device)  # B C Y X
            semantic_masks = sample["semantic_mask"].to(self.device)  # B C Y X
            # semantic_masks.squeeze_(1)  #FIXME B Z Y X (loss expects this format) JO But it doesn't match output
            output = self.model(images)  # B C Y X
            if isinstance(self.criterion, loss_utils.TverskyLoss):
                loss = self.criterion(output, semantic_masks.long())  # B C Y X
            elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
                loss = self.criterion(
                    output, semantic_masks.float()
                )  # B C Y X
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_meter.update(loss.item())

        return loss_meter.avg

    def val(self):
        loss_meter = AverageMeter()
        average_precision_meter = AverageMeter()
        average_dice_meter = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(tqdm(self.val_dataset_it)):
                images = sample["image"].to(self.device)  # B C Y X
                semantic_masks = sample["semantic_mask"].to(self.device)  # B C Y X
                # semantic_masks.squeeze_(1)  #FIXME B Z Y X (loss expects this format) JO But it doesn't match output
                output = self.model(images)  # B C Y X
                if isinstance(self.criterion, loss_utils.TverskyLoss):
                    loss = self.criterion(output, semantic_masks.long())  # B C Y X
                elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    loss = self.criterion(
                        output, semantic_masks.float()
                    )  # B C Y X
                loss_meter.update(loss.item())

                # Compute segmentation metrics
                if self.model_dict["kwargs"]["num_classes"] == 1:
                    # "0-1" activation applys sigmoid and thresholds above 0.5
                    # aka "0-1" performs activation: (torch.sigmoid(output) > 0.5).float()
                    val_metrics = BinaryMetrics(activation="0-1")
                    acc, dice, prec, spec, rec = val_metrics(semantic_masks.cpu().detach(), output.cpu().detach())
                else:
                    val_metrics = SegmentationMetrics(activation=None)
                    raise NotImplementedError("No activation defined for multiclass problem")
                average_precision_meter.update(prec.item())
                average_dice_meter.update(dice.item())

                # # bin_metric = BinaryMetrics(activation='0-1') # Does a sigmoid and threshold activation
                # # acc, dice, prec, spec, rec = bin_metric(semantic_masks[:, 0, ...], output)
                # # average_precision_meter.update(dice)
                # for b in range(output.shape[0]):
                #     output_softmax = torch.sigmoid(output[b])
                #     prediction_fg = output_softmax[0, ...].cpu().detach().numpy()
                #     pred_fg_thresholded = (prediction_fg > 0.5).astype(int)
                #     acc, dice, prec, spec, rec = binary_sem_seg_metrics(
                #         y_true=semantic_masks[b, 0, ...].cpu().detach().numpy(),
                #         y_pred=pred_fg_thresholded,
                #     )
                #     # sc = matching_dataset(y_pred=[pred_fg_thresholded], y_true=[semantic_masks[b, 0, ...].cpu().detach().numpy()])
                #     # instance_map, _ = ndimage.label(pred_fg_thresholded)
                #     #     # sc = matching_dataset(y_pred=[instance_map], y_true=[instance[b, 0, ...].cpu().detach().numpy()],
                #     #     #                     thresh=0.5, show_progress=False)
                #     average_precision_meter.update(dice)

        return loss_meter.avg, average_precision_meter.avg, average_dice_meter.avg


def main(
    data_dir: str,
    output_dir: str,
    n_channels: int = 1,
    n_classes: int = 1,
    n_levels: int = 3,
    in_size: tuple = (128, 128),
    init_lr: float = 1e-3,
    n_epochs: int = 15,
    batch_size: int = 8,
    save: bool = True,
):
    """
    Args:
        data_dir (str): Directory containing "train", "val", "test" folders
        output_dir (str): Directory where training results will be saved
        n_channels (int): Number of channels in input image (idk if it can be other than 1)
        n_classes (int): Number of classes for segmentation
        n_levels (int): Number of levels in the U-Net
        in_size (tuple): Size of images after resizing prior to input into network
        init_lr (float): Initial learn rate
        n_epochs (int): Number of training/validation epochs during training
        batch_size (int): Batchsize of training images in network
        save (bool): Whether to save training results
    """
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
        save=save,
    )
    train_dataset_dict = cfg.TRAIN_DATASET_DICT
    print("Training dictionary:")
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
    model_trainer = Trainer(
        train_dataset_dict, val_dataset_dict, test_dataset_dict, model_dict, config_dict
    )
    model_trainer.begin_training()


if __name__ == "__main__":
    from src.definitions import ROOT_DIR, DATA_DIR

    oct_data_dir = f"{DATA_DIR}/processed/OCT_scans_128x128"
    results_dir = f"{ROOT_DIR}/src/models/dl4mia_tissue_unet/results"
    main(oct_data_dir, results_dir, n_epochs=5, save=False)
