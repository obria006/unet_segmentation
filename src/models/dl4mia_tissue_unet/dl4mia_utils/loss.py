import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Binary dice loss for semantic segmentation. This code is reworked from
    these GitHub repos:
        - https://github.com/qubvel/segmentation_models.pytorch
        - https://github.com/BloodAxe/pytorch-toolbelt

    """

    def __init__(
        self,
        mode: str,
        from_logits: bool = True,
        log_loss: bool = False,
        smooth: float = 0.0,
        eps: float = 1e-7,
    ):
        """
        Args:
            mode (str): Loss mode 'binary', 'multiclass' or 'multilabel'
            log_loss (bool): If True, the loss is computed as `-log(dice_coeff)`,
                otherwise `1 - dice_coeff`.
            from_logits (bool): If True, assumes y_pred are raw logits.
            smooth (float): Smoothness constant for dice coefficient.
            eps (float): For numerical stability to avoid zero division error.

        """
        super(DiceLoss, self).__init__()

        assert mode in ["binary", "multiclass", "multilabel"]
        self.mode = mode
        self.from_logits = from_logits
        self.log_loss = log_loss
        self.smooth = smooth
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Args:
            y_pred (torch.Tensor): Of shape (B, C, H, W).
            y_true (torch.Tensor): Of shape (B, C, H, W).

        Returns:
            torch.Tensor: The loss.
        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            if self.mode == "multiclass":
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == "binary":
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

        if self.mode == "multiclass":
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1)  # N, C, H*W

        if self.mode == "multilabel":
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self._compute_score(
            y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss = loss * mask.to(loss.dtype)

        return self._reduction(loss)

    def _reduction(self, loss):
        return loss.mean()

    def _compute_score(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, smooth=0.0, eps=1e-7, dims=()
    ):
        assert y_pred.size() == y_true.size()

        intersection = torch.sum(y_pred * y_true, dim=dims)
        cardinality = torch.sum(y_pred + y_true, dim=dims)

        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(
            eps
        )

        return dice_score


class TverskyLoss(DiceLoss):
    """
    This code is reworked from this GitHub repo:
        - https://github.com/qubvel/segmentation_models.pytorch
        - https://github.com/BloodAxe/pytorch-toolbelt

    Tversky loss for semantic segmentation. Notice this class inherits
    `DiceLoss` and adds a weight to the value of each TP and FP given by
    constants alpha and beta. With alpha == beta == 0.5, this loss becomes
    equal to the Dice loss. `y_pred` and `y_true` must be torch tensors of
    shape (B, C, H, W).

    """

    def __init__(
        self,
        mode: str,
        from_logits: bool = True,
        log_loss: bool = False,
        smooth: float = 0.0,
        eps: float = 1e-7,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
    ):
        """
        Args:
            mode (str): Loss mode 'binary', 'multiclass' or 'multilabel'
            from_logits (bool): If True, assumes y_pred are raw logits.
            log_loss (bool): If True, the loss is computed as `-log(dice_coeff)`,
                otherwise `1 - dice_coeff`.

            smooth (float): Smoothness constant for dice coefficient.
            eps (float): For numerical stability to avoid zero division error.
            alpha (float): Weight constant that penalize model for FPs.
            beta (float): Weight constant that penalize model for FNs.
            gamma (float): Constant that squares the error function. Defaults to `1.0`.

        """
        super(TverskyLoss, self).__init__(mode, from_logits, log_loss, smooth, eps)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _reduction(self, loss):
        return loss.mean() ** self.gamma

    def _compute_score(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        smooth: float = 0.0,
        eps: float = 1e-7,
        dims=(),
    ):
        assert y_pred.size() == y_true.size()

        intersection = torch.sum(y_pred * y_true, dim=dims)
        fp = torch.sum(y_pred * (1.0 - y_true), dim=dims)
        fn = torch.sum((1 - y_pred) * y_true, dim=dims)

        tversky_score = (intersection + smooth) / (
            intersection + self.alpha * fp + self.beta * fn + smooth
        ).clamp_min(eps)

        return tversky_score
