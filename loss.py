import torch.nn as nn
import torch.nn.functional as F
import torch


class JointsKLLoss(nn.Module):
    """
    KL Divergence for keypoint detection proposed by
    `Regressive Domain Adaptation for Unsupervised Keypoint Detection <https://arxiv.org/abs/2103.06175>`_.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Inputs:
        - output (tensor): heatmap predictions
        - target (tensor): heatmap labels
        - target_weight (tensor): whether the keypoint is visible. All keypoint is visible if None. Default: None.

    Shape:
        - output: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - target: :math:`(minibatch, K, H, W)`.
        - target_weight: :math:`(minibatch, K)`.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, K)`.

    """
    def __init__(self, reduction='mean'):
        super(JointsKLLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.reduction = reduction


    def forward(self, output, target, target_weight=None):
        B, K, _, _ = output.shape
        output = output.to(torch.float32)
        target = target.to(torch.float32)
        heatmaps_pred = output.reshape((B, K, -1))
        heatmaps_pred = F.log_softmax(heatmaps_pred, dim=-1)

        heatmaps_gt = target.reshape((B, K, -1))
        heatmaps_gt = heatmaps_gt

        heatmaps_gt = heatmaps_gt / heatmaps_gt.sum(dim=-1, keepdims=True)

        loss = self.criterion(heatmaps_pred, heatmaps_gt).sum(dim=-1)

        if target_weight is not None:
             loss = loss * target_weight.view((B, K))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss.mean(dim=-1)

class AI_MSELoss(nn.Module):

    def __init__(self):
        super(AI_MSELoss, self).__init__()

    def forward(self, y_s, y_t, y_gt, phi, alpha):

        loss = torch.mean(torch.square(y_s - y_t)*alpha+torch.square(y_s - y_gt)*phi*(1-alpha))

        return loss