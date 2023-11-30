import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Any, Tuple
import numpy as np
import torch.nn as nn
from torch.autograd import Function
import torch

train_on_gpu = torch.cuda.is_available()
DEVICE = torch.device('cuda' if train_on_gpu else 'cpu')

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


class PseudoLabelGenerator2d(nn.Module):
    """
    Generate ground truth heatmap and ground false heatmap from a prediction.

    Args:
        num_keypoints (int): Number of keypoints
        height (int): height of the heatmap. Default: 64
        width (int): width of the heatmap. Default: 64
        sigma (int): sigma parameter when generate the heatmap. Default: 2

    Inputs:
        - y: predicted heatmap

    Outputs:
        - ground_truth: heatmap conforming to Gaussian distribution
        - ground_false: ground false heatmap

    Shape:
        - y: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - ground_truth: :math:`(minibatch, K, H, W)`
        - ground_false: :math:`(minibatch, K, H, W)`
    """
    def __init__(self, num_keypoints, height=64, width=64, sigma=2):
        super(PseudoLabelGenerator2d, self).__init__()
        self.height = height
        self.width = width
        self.sigma = sigma

        heatmaps = np.zeros((width, height, height, width), dtype=np.float32)

        tmp_size = sigma * 6
        for mu_x in range(width):
            for mu_y in range(height):
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

                # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], width) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], height) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], width)
                img_y = max(0, ul[1]), min(br[1], height)

                heatmaps[mu_x][mu_y][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        self.heatmaps = heatmaps
        self.false_matrix = 1. - np.eye(num_keypoints, dtype=np.float32)

    def forward(self, y):
        B, K, H, W = y.shape
        y = y.detach()
        preds,maxvals = get_max_preds(y.cpu().numpy())  # B x K x (x, y)
       
        preds = preds.reshape(-1, 2).astype(np.int)
        ground_truth = self.heatmaps[preds[:, 0], preds[:, 1], :, :].copy().reshape(B, K, H, W).copy()

        ground_false = ground_truth.reshape(B, K, -1).transpose((0, 2, 1))

        ground_false = ground_false.dot(self.false_matrix).clip(max=1., min=0.).transpose((0, 2, 1)).reshape(B, K, H, W).copy()

        return torch.from_numpy(ground_truth).to(y.device), torch.from_numpy(ground_false).to(y.device)


class RegressionDisparity(nn.Module):
    """
    Regression Disparity proposed by `Regressive Domain Adaptation for Unsupervised Keypoint Detection (CVPR 2021) <https://arxiv.org/abs/2103.06175>`_.

    Args:
        pseudo_label_generator (PseudoLabelGenerator2d): generate ground truth heatmap and ground false heatmap
          from a prediction.
        criterion (torch.nn.Module): the loss function to calculate distance between two predictions.

    Inputs:
        - y: output by the main head
        - y_adv: output by the adversarial head
        - weight (optional): instance weights
        - mode (str): whether minimize the disparity or maximize the disparity. Choices includes ``min``, ``max``.
          Default: ``min``.

    Shape:
        - y: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - y_adv: :math:`(minibatch, K, H, W)`
        - weight: :math:`(minibatch, K)`.
        - Output: depends on the ``criterion``.

    Examples::

        >>> num_keypoints = 5
        >>> batch_size = 10
        >>> H = W = 64
        >>> pseudo_label_generator = PseudoLabelGenerator2d(num_keypoints)
        >>> from tllibvision.models.keypoint_detection.loss import JointsKLLoss
        >>> loss = RegressionDisparity(pseudo_label_generator, JointsKLLoss())
        >>> # output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # adversarial output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # minimize regression disparity on source domain
        >>> output = loss(y_s, y_s_adv, mode='min')
        >>> # maximize regression disparity on target domain
        >>> output = loss(y_t, y_t_adv, mode='max')
    """
    def __init__(self, pseudo_label_generator: PseudoLabelGenerator2d, criterion: nn.Module):
        super(RegressionDisparity, self).__init__()
        self.criterion = criterion
        self.pseudo_label_generator = pseudo_label_generator

    def forward(self, y, y_adv, weight=None, mode='min'):
        assert mode in ['min', 'max']
        ground_truth, ground_false = self.pseudo_label_generator(y.detach())
        if mode == 'min':
            return self.criterion(y_adv, ground_truth, weight)
        else:
            return self.criterion(y_adv, ground_false, weight)

def softargmax2d(input, beta=1000):
    *_, h, w = input.shape

    input = beta*input.reshape(*_, h * w)
    input = F.softmax( input, dim=-1)

    indices_c, indices_r = np.meshgrid(
        np.linspace(0, 1, w),
        np.linspace(0, 1, h),
        indexing='xy'
    )

    indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w))).to(DEVICE)
    indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w))).to(DEVICE)

    result_r = torch.sum((h - 1) * input * indices_r, dim=-1)
    result_c = torch.sum((w - 1) * input * indices_c, dim=-1)

    result = torch.stack([result_r, result_c], dim=-1)

    return result

class CConv2d(nn.Module):
    """
    Class of complex valued convolutional layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.real_conv = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding,
                                   stride=self.stride)

        self.im_conv = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 kernel_size=self.kernel_size,
                                 padding=self.padding,
                                 stride=self.stride)

        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)

    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)

        output = torch.stack([c_real, c_im], dim=-1)
        return output


class CConvTranspose2d(nn.Module):
    """
      Class of complex valued dilation convolutional layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()

        self.in_channels = in_channels

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride

        self.real_convt = nn.ConvTranspose2d(in_channels=self.in_channels,
                                             out_channels=self.out_channels,
                                             kernel_size=self.kernel_size,
                                             output_padding=self.output_padding,
                                             padding=self.padding,
                                             stride=self.stride)

        self.im_convt = nn.ConvTranspose2d(in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=self.kernel_size,
                                           output_padding=self.output_padding,
                                           padding=self.padding,
                                           stride=self.stride)

        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)

    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)

        output = torch.stack([ct_real, ct_im], dim=-1)
        return output


class CBatchNorm2d(nn.Module):
    """
    Class of complex valued batch normalization layer
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                     affine=self.affine, track_running_stats=self.track_running_stats)
        self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                   affine=self.affine, track_running_stats=self.track_running_stats)

    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        n_real = self.real_b(x_real)
        n_im = self.im_b(x_im)

        output = torch.stack([n_real, n_im], dim=-1)
        return output


class BasicBlock(nn.Module):

    def __init__(self, filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45, padding=(1, 1)):
        super().__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cconv = CConv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                             kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)

        self.cbn = CBatchNorm2d(num_features=self.out_channels)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        conved = self.cconv(x)
        normed = self.cbn(conved)
        acted = self.leaky_relu(normed)

        return acted


class Maxpoolings(nn.Module):
    """
    Class of upsample block
    """

    def __init__(self):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)


    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        n_real = self.maxpool(x_real)
        n_im = self.maxpool(x_im)

        output = torch.stack([n_real, n_im], dim=-1)
        return output

class Upooling(nn.Module):
    """
    Class of upsample block
    """

    def __init__(self):
        super().__init__()

        self.upooling = nn.Upsample(scale_factor=2)


    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        n_real = self.upooling(x_real)
        n_im = self.upooling(x_im)

        output = torch.stack([n_real, n_im], dim=-1)
        return output

def Connect(x1,x2):
    x1_real = x1[..., 0]
    x1_im = x1[..., 1]

    x2_real = x2[..., 0]
    x2_im = x2[..., 1]

    x_real = torch.cat((x1_real, x2_real), dim=1)
    x_im =  torch.cat((x1_im, x2_im), dim=1)
    output = torch.stack([x_real,x_im], dim=-1)
    return output


class Decoder(nn.Module):
    """
    Class of downsample block
    """

    def __init__(self, filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45,
                 output_padding=(0, 0), padding=(0, 0), last_layer=False):
        super().__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding

        self.last_layer = last_layer

        self.cconvt = CConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                       kernel_size=self.filter_size, stride=self.stride_size,
                                       output_padding=self.output_padding, padding=self.padding)

        self.cbn = CBatchNorm2d(num_features=self.out_channels)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):

        conved = self.cconvt(x)

        if not self.last_layer:
            normed = self.cbn(conved)
            output = self.leaky_relu(normed)
        else:
            m_phase = conved / (torch.abs(conved) + 1e-8)
            m_mag = torch.tanh(torch.abs(conved))
            output = m_phase * m_mag

        return output


class feature_extrator(nn.Module):
    """
    Deep Complex U-Net class of the model.
    """

    def __init__(self):
        super().__init__()

        # downsampling/encoding

        self.encoder0 = BasicBlock(filter_size=3, stride_size=1, in_channels=10, out_channels=32)
        self.encoder1 = BasicBlock(filter_size=3, stride_size=1, in_channels=32, out_channels=32)
        self.max1 = Maxpoolings()

        self.encoder2 = BasicBlock(filter_size=3, stride_size=1, in_channels=32, out_channels=64)
        self.encoder3 = BasicBlock(filter_size=3, stride_size=1, in_channels=64, out_channels=64)
        self.max2 = Maxpoolings()

        self.encoder4 = BasicBlock(filter_size=3, stride_size=1, in_channels=64, out_channels=128)
        self.encoder5 = BasicBlock(filter_size=3, stride_size=1, in_channels=128, out_channels=128)
        self.max3 = Maxpoolings()

        self.encoder6 = BasicBlock(filter_size=3, stride_size=1, in_channels=128, out_channels=256)
        self.encoder7 = BasicBlock(filter_size=3, stride_size=1, in_channels=256, out_channels=256)

        self.up1 = Upooling()
        self.decoder0 = BasicBlock(filter_size=3, stride_size=1, in_channels=256, out_channels=128)
        self.decoder1 = BasicBlock(filter_size=3, stride_size=1, in_channels=128, out_channels=128)

        self.up2 = Upooling()
        self.decoder2 = BasicBlock(filter_size=3, stride_size=1, in_channels=256, out_channels=64)
        self.decoder3 = BasicBlock(filter_size=3, stride_size=1, in_channels=64, out_channels=64)

        self.up3 = Upooling()
        self.decoder4 = BasicBlock(filter_size=3, stride_size=1, in_channels=128, out_channels=32)
        self.decoder5 = BasicBlock(filter_size=3, stride_size=1, in_channels=32, out_channels=32)

        self.decoder6 = BasicBlock(filter_size=3, stride_size=1, in_channels=64, out_channels=32)
        self.decoder7 = BasicBlock(filter_size=3, stride_size=1, in_channels=32, out_channels=32)

        self.up4 = Upooling()
        self.decoder8 = BasicBlock(filter_size=3, stride_size=1, in_channels=32, out_channels=32)


    def forward(self, x):
        
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        m1 = self.max1(e1)

        e2 = self.encoder2(m1)
        e3 = self.encoder3(e2)
        m2 = self.max2(e3)

        e4 = self.encoder4(m2)
        e5 = self.encoder5(e4)
        m3 = self.max3(e5)

        e6 = self.encoder6(m3)
        e7 = self.encoder7(e6)

        up1 = self.up1(e7)
        d0 = self.decoder0(up1)
        d1 = self.decoder1(d0)
        c1 = Connect(e5, d1)

        up2 = self.up2(c1)
        d2 = self.decoder2(up2)
        d3 = self.decoder3(d2)
        c2 = Connect(e3, d3)

        up3 = self.up3(c2)
        d4 = self.decoder4(up3)
        d5 = self.decoder5(d4)

        c3 = Connect(e1, d5)
        d6 = self.decoder6(c3)
        d7 = self.decoder7(d6)

        up4 = self.up4(d7)
        d8 = self.decoder8(up4)
        d8 = d8.reshape((-1, 64, 80, 80))

        return d8


class GradientFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output * ctx.coeff, None


class WarmStartGradientLayer(nn.Module):
    """Warm Start Gradient Layer :math:`\mathcal{R}(x)` with warm start

        The forward and backward behaviours are:

        .. math::
            \mathcal{R}(x) = x,

            \dfrac{ d\mathcal{R}} {dx} = \lambda I.

        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:

        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo

        where :math:`i` is the iteration step.

        Parameters:
            - **alpha** (float, optional): :math:`α`. Default: 1.0
            - **lo** (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            - **hi** (float, optional): Final value of :math:`\lambda`. Default: 1.0
            - **max_iters** (int, optional): :math:`N`. Default: 1000
            - **auto_step** (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


class OCHID_Fi(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_extractor = feature_extrator()
        self.head = self._make_head(3, 64, 21)
        self.head_depth = self._make_head(3, 64, 21)
        self.head_adv = self._make_head(3, 64, 21)
        self.head_depth_adv = self._make_head(3, 64, 21)
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000,
                                               auto_step=False)
        self.softmax = nn.Softmax2d()

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.LeakyReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x, mode=None):
        features = self.feature_extractor(x)
        xy_map = self.head(features)  #xy map
        latent_depth = self.head_depth(features)  # z map
        prob_2d = self.softmax(xy_map)
        xy = softargmax2d(prob_2d)
        depth_map = latent_depth*prob_2d
        depth = (torch.sum(depth_map,(2,3)).unsqueeze(dim=-1))/(80*80)
        if mode == 'pre_train':
            return xy, depth
        gl = self.gl_layer(features)

        xy_map_adv = self.head_adv(gl)
        latent_depth_adv = self.head_depth_adv(gl)
        prob_2d_adv = self.softmax(xy_map_adv)
        xy_adv = softargmax2d(prob_2d_adv)
        depth_map_adv = latent_depth_adv*prob_2d_adv
        depth_adv = (torch.sum(depth_map_adv,(2,3)).unsqueeze(dim=-1))/(80*80)
        
        if mode == 'train':
            return xy, depth, xy_map, xy_map_adv, depth_adv
        else:
            return xy, depth

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()