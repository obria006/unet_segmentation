import numpy as np
import torch
import torch.nn as nn


class SegmentationMetrics(object):
    r"""(from: https://github.com/hsiangyuzhao/Segmentation-Metrics-PyTorch/blob/master/metric.py)
    
    Calculate common metrics in semantic segmentation to evalueate model preformance.

    Supported metrics: Pixel accuracy, Dice Coeff, precision score and recall score.

    Pixel accuracy measures how many pixels in a image are predicted correctly.

    Dice Coeff is a measure function to measure similarity over 2 sets, which is usually used to
    calculate the similarity of two samples. Dice equals to f1 score in semantic segmentation tasks.

    It should be noted that Dice Coeff and Intersection over Union are highly related, so you need
    NOT calculate these metrics both, the other can be calcultaed directly when knowing one of them.

    Precision describes the purity of our positive detections relative to the ground truth. Of all
    the objects that we predicted in a given image, precision score describes how many of those objects
    actually had a matching ground truth annotation.

    Recall describes the completeness of our positive predictions relative to the ground truth. Of
    all the objected annotated in our ground truth, recall score describes how many true positive instances
    we have captured in semantic segmentation.

    Args:
        eps: float, a value added to the denominator for numerical stability.
            Default: 1e-5

        average: bool. Default: ``True``
            When set to ``True``, average Dice Coeff, precision and recall are
            returned. Otherwise Dice Coeff, precision and recall of each class
            will be returned as a numpy array.

        ignore_background: bool. Default: ``True``
            When set to ``True``, the class will not calculate related metrics on
            background pixels. When the segmentation of background pixels is not
            important, set this value to ``True``.

        activation: [None, 'none', 'softmax' (default), 'sigmoid', '0-1']
            This parameter determines what kind of activation function that will be
            applied on model output.

    Input:
        y_true: :math:`(N, H, W)`, torch tensor, where we use int value between (0, num_class - 1)
        to denote every class, where ``0`` denotes background class.
        y_pred: :math:`(N, C, H, W)`, torch tensor.

    Examples::
        >>> metric_calculator = SegmentationMetrics(average=True, ignore_background=True)
        >>> pixel_accuracy, dice, precision, recall = metric_calculator(y_true, y_pred)
    """

    def __init__(
        self, eps=1e-5, average=True, ignore_background=True, activation="0-1"
    ):
        self.eps = eps
        self.average = average
        self.ignore = ignore_background
        self.activation = activation

    @staticmethod
    def _one_hot(gt, pred, class_num):
        # transform sparse mask into one-hot mask
        # shape: (B, H, W) -> (B, C, H, W)
        input_shape = tuple(gt.shape)  # (N, H, W, ...)
        new_shape = (input_shape[0], class_num) + input_shape[1:]
        one_hot = torch.zeros(new_shape).to(pred.device, dtype=torch.float)
        target = one_hot.scatter_(1, gt.unsqueeze(1).long().data, 1.0)
        return target

    @staticmethod
    def _get_class_data(gt_onehot, pred, class_num):
        # perform calculation on a batch
        # for precise result in a single image, plz set batch size to 1
        matrix = np.zeros((3, class_num))

        # calculate tp, fp, fn per class
        for i in range(class_num):
            # pred shape: (N, H, W)
            class_pred = pred[:, i, :, :]
            # gt shape: (N, H, W), binary array where 0 denotes negative and 1 denotes positive
            class_gt = gt_onehot[:, i, :, :]

            pred_flat = class_pred.contiguous().view(
                -1,
            )  # shape: (N * H * W, )
            gt_flat = class_gt.contiguous().view(
                -1,
            )  # shape: (N * H * W, )

            tp = torch.sum(gt_flat * pred_flat)
            fp = torch.sum(pred_flat) - tp
            fn = torch.sum(gt_flat) - tp

            matrix[:, i] = tp.item(), fp.item(), fn.item()

        return matrix

    def _calculate_multi_metrics(self, gt, pred, class_num):
        # calculate metrics in multi-class segmentation
        matrix = self._get_class_data(gt, pred, class_num)
        if self.ignore:
            matrix = matrix[:, 1:]

        # tp = np.sum(matrix[0, :])
        # fp = np.sum(matrix[1, :])
        # fn = np.sum(matrix[2, :])
        pixel_acc = (np.sum(matrix[0, :]) + self.eps) / (
            np.sum(matrix[0, :]) + np.sum(matrix[1, :]) + self.eps
        )
        dice = (2 * matrix[0] + self.eps) / (
            2 * matrix[0] + matrix[1] + matrix[2] + self.eps
        )
        precision = (matrix[0] + self.eps) / (matrix[0] + matrix[1] + self.eps)
        recall = (matrix[0] + self.eps) / (matrix[0] + matrix[2] + self.eps)

        if self.average:
            dice = np.average(dice)
            precision = np.average(precision)
            recall = np.average(recall)

        return pixel_acc, dice, precision, recall

    def __call__(self, y_true, y_pred):
        if len(y_true.shape) == 4:
            assert y_true.shape[0] == y_pred.shape[0], f"'true' batch {y_true.shape} mismatch 'pred' batch {y_pred.shape}"
            assert y_true.shape[-2:] == y_pred.shape[-2:], f"'true' H, W {y_true.shape} mismatch 'pred' H, W {y_pred.shape}"
            assert y_true.shape[1] == 1, f"Invalid 'true' shape {y_true.shape}. Must be (B, 1, H, W)"
            y_true = torch.clone(y_true).detach()[:,0,:,:]
        assert len(y_true.shape) == 3, f"Invalid 'true' shape: {y_true.shape}. Must be (B, H, W)"
        assert len(y_pred.shape) == 4, f"Invalid 'pred' shape: {y_pred.shape}. Must be (B, C, H, W)"
        class_num = y_pred.size(1)

        if self.activation in [None, "none"]:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "softmax":
            activation_fn = nn.Softmax(dim=1)
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            pred_argmax = torch.argmax(y_pred, dim=1)
            activated_pred = self._one_hot(pred_argmax, y_pred, class_num)
        else:
            raise NotImplementedError("Not a supported activation!")

        gt_onehot = self._one_hot(y_true, y_pred, class_num)
        pixel_acc, dice, precision, recall = self._calculate_multi_metrics(
            gt_onehot, activated_pred, class_num
        )
        return pixel_acc, dice, precision, recall


class BinaryMetrics:
    r"""Calculate common metrics in binary cases.
    In binary cases it should be noted that y_pred shape shall be like (N, 1, H, W), or an assertion
    error will be raised.
    Also this calculator provides the function to calculate specificity, also known as true negative
    rate, as specificity/TPR is meaningless in multiclass cases.
    """

    def __init__(self, eps=1e-5, activation="0-1"):
        self.eps = eps
        self.activation = activation

    def _calculate_overlap_metrics(self, gt, pred):
        output = pred.view(
            -1,
        )
        target = gt.view(
            -1,
        ).float()
        assert len(torch.unique(target)) <= 2, "GT must be binary"

        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP
        fn = torch.sum((1 - output) * target)  # FN
        tn = torch.sum((1 - output) * (1 - target))  # TN

        pixel_acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
        dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps)
        recall = (tp + self.eps) / (tp + fn + self.eps)
        specificity = (tn + self.eps) / (tn + fp + self.eps)

        return pixel_acc, dice, precision, specificity, recall

    def __call__(self, y_true, y_pred):
        if len(y_true.shape) == 4:
            assert y_true.shape == y_pred.shape, f"'true' shape {y_true.shape} mismatch 'pred' shape {y_pred.shape}"
            assert y_true.shape[1] == 1, f"Invalid 'true' shape {y_true.shape} for binary. Must be (B, 1, H, W)"
            y_true = torch.clone(y_true).detach()[:,0,:,:]
        assert len(y_true.shape) == 3, f"Invalid 'true' shape: {y_true.shape}. Must be (B, H, W)"
        assert len(y_pred.shape) == 4, f"Invalid 'pred' shape: {y_pred.shape}. Must be (B, 1, H, W)"
        assert y_pred.shape[1] == 1, f"Invalid 'pred' shape: {y_pred.shape}. Must be (B, 1, H, W)"
        # y_true: (N, H, W)
        # y_pred: (N, 1, H, W)
        if self.activation in [None, "none"]:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            sigmoid_pred = nn.Sigmoid()(y_pred)
            activated_pred = (sigmoid_pred > 0.5).float().to(y_pred.device)
        else:
            raise NotImplementedError("Not a supported activation!")

        assert activated_pred.shape[1] == 1, (
            "Predictions must contain only one channel"
            " when performing binary segmentation"
        )
        (
            pixel_acc,
            dice,
            precision,
            specificity,
            recall,
        ) = self._calculate_overlap_metrics(
            y_true.to(y_pred.device, dtype=torch.float), activated_pred
        )
        return [pixel_acc, dice, precision, specificity, recall]


def binary_sem_seg_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-5):
    """Takes a binary ground truth ndarray and the associated binary
    prediction ndarray and returns the segmentation metrics: accuracy, dice
    (same as f1 for binary semantic segmentation), precision, specificity and
    recall.

    Arguments:
        y_true (np.ndarray): Binary ground truth.
        y_pred (np.ndarray): Binary prediction.
        eps (float): Small epsilon to avoid division by 0.

    Returns:
        accuracy, dice, precision, specificity and recall of prediction
    """
    assert y_true.shape == y_pred.shape, "Images for metrics must be same shape"
    assert (
        len(np.unique(y_true)) <= 2
    ), f"Ground truth image for binary metric must be binary. Invalid values: {np.unique(y_true)}"
    assert (
        len(np.unique(y_pred)) <= 2
    ), f"Prediction image for binary metric must be binary. Invalid values: {np.unique(y_pred)}"
    assert (
        np.amin(y_true) >= 0 and np.amax(y_true) <= 1
    ), f"Ground truth image for binary metric must be 0-1 valued. Invalid values: {np.unique(y_true)}"
    assert (
        np.amin(y_pred) >= 0 and np.amax(y_pred) <= 1
    ), f"Prediction image for binary metric must be 0-1 valued. Invalid values: {np.unique(y_pred)}"

    tp = np.sum(y_pred * y_true)  # TP
    fp = np.sum(y_pred * (1 - y_true))  # FP
    fn = np.sum((1 - y_pred) * y_true)  # FN
    tn = np.sum((1 - y_pred) * (1 - y_true))  # TN

    pixel_acc, dice, precision, specificity, recall = metrics_from_confusion_nums(
        tp=tp, fp=fp, fn=fn, tn=tn, eps=eps
    )

    return pixel_acc, dice, precision, specificity, recall


def metrics_from_confusion_nums(tp: int, fp: int, fn: int, tn: int, eps: float = 1e-5):
    """
    Computes metrics from the confusion matrix numbers: true positives (tp),
    false positives (fp), false negatives (fn), and true negatives (tn).

    Args:
        tp (int): True positives
        fp (int): False positives
        fn (int): False negatives
        tn (int): True negatives

    Returns:
        accuracy, f1, precision, specificity and recall
    """

    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return acc, f1, precision, specificity, recall
