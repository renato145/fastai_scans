from fastai.basics import *
from fastai.layers import SelfAttention

__all__ = ['SegCrossEntropy', 'ParallelLoss', 'AdaptiveConcatPool3d', 'Upsample3d', 'Upsample3dLike', 'upsample3d_like',
           'pixel_shuffle_nd', 'icnr3d', 'PixelShuffle3d_ICNR',
           'batchnorm_3d', 'conv3d', 'conv3d_trans', 'conv_layer3d', 'res_block3d',
           'ResLayer3d', 'Apply2dConv', 'ReshapeSeg2d']

class SegCrossEntropy(nn.CrossEntropyLoss):
    def forward(self, input, target):
        n,c,*_ = input.shape
        return super().forward(input.view(n, c, -1), target.view(n, -1))

class ParallelLoss(nn.Module):
    def __init__(self, diagnosis_weight=0.5):
        super().__init__()
        self.diagnosis_weight = diagnosis_weight
        self.seg_loss = SegCrossEntropy()
        self.lbl_loss = nn.CrossEntropyLoss()
        self.metric_names = ['seg_loss', 'cls_loss']
        
    def forward(self, input, targ_seg, targ_lbl):
        seg, lbl = input
        L_seg = self.seg_loss(seg, targ_seg)
        L_lbl = self.lbl_loss(lbl, targ_lbl)
        self.metrics = dict(zip(self.metric_names, [L_seg, L_lbl]))
        return self.diagnosis_weight*L_lbl + (1-self.diagnosis_weight)*L_seg

class AdaptiveConcatPool3d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or 1
        self.ap, self.mp = nn.AdaptiveAvgPool3d(sz), nn.AdaptiveMaxPool3d(sz)
    
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class Upsample3d(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x): return F.interpolate(x, self.shape, mode='nearest')
    
class Upsample3dLike(Upsample3d):
    def __init__(self, target):
        super().__init__(target.shape[-3:])

def upsample3d_like(x, target): return F.interpolate(x, target.shape[-3:], mode='nearest')

def pixel_shuffle_nd(input, upscale_factor):
    input_size = list(input.size())
    dimensionality = len(input_size) - 2

    input_size[1] //= (upscale_factor ** dimensionality)
    output_size = [dim * upscale_factor for dim in input_size[2:]]

    input_view = input.contiguous().view(
        input_size[0], input_size[1],
        *(([upscale_factor] * dimensionality) + input_size[2:])
    )

    indicies = list(range(2, 2 + 2 * dimensionality))
    indicies = indicies[1::2] + indicies[0::2]

    shuffle_out = input_view.permute(0, 1, *(indicies[::-1])).contiguous()
    
    return shuffle_out.view(input_size[0], input_size[1], *output_size)

def icnr3d(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function."
    ni,nf,h,w,d = x.shape # height, width, depth
    ni2 = int(ni/(scale**3))
    k = init(torch.zeros([ni2,nf,h,w,d])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**3)
    k = k.contiguous().view([nf,ni,h,w,d]).transpose(0, 1)
    x.data.copy_(k)

class PixelShuffle3d_ICNR(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."
    def __init__(self, ni:int, nf:int=None, scale:int=2, norm_type=NormType.Weight, leaky:float=None, **kwargs):
        super().__init__()
        nf = ifnone(nf, ni)
        self.scale = scale
        self.conv = conv_layer3d(ni, nf*(scale**3), ks=1, norm_type=norm_type, use_activ=False, **kwargs)
        icnr3d(self.conv[0].weight)
        self.relu = relu(True, leaky=leaky)

    def forward(self,x): return pixel_shuffle_nd(self.relu(self.conv(x)), self.scale)

def batchnorm_3d(nf:int, norm_type:NormType=NormType.Batch):
    "A batchnorm3d layer with `nf` features initialized depending on `norm_type`."
    bn = nn.BatchNorm3d(nf)
    with torch.no_grad():
        bn.bias.fill_(1e-3)
        bn.weight.fill_(0. if norm_type==NormType.BatchZero else 1.)
    return bn

def conv3d(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias=False, norm_type:Optional[NormType]=NormType.Batch,
           init:LayerFunc=nn.init.kaiming_normal_, **kwargs) -> nn.Conv3d:
    "Create and initialize `nn.Conv3d` layer. `padding` defaults to `ks//2`."
    if padding is None: padding = ks//2
    conv = init_default(nn.Conv3d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias, **kwargs), init)
    if   norm_type==NormType.Weight:   conv = weight_norm(conv)
    elif norm_type==NormType.Spectral: conv = spectral_norm(conv)
    return conv

def conv3d_trans(ni:int, nf:int, ks:int=2, stride:int=2, padding:int=0, bias=False) -> nn.ConvTranspose3d:
    "Create `nn.ConvTranspose3d` layer."
    return nn.ConvTranspose3d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias)

def conv_layer3d(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias:bool=None,
                 norm_type:Optional[NormType]=NormType.Batch, use_activ:bool=True, leaky:float=None,
                 separable:bool=False, init:Callable=nn.init.kaiming_normal_, self_attention:bool=False):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None: padding = (ks-1)//2
    bn = norm_type in (NormType.Batch, NormType.BatchZero)
    if bias is None: bias = not bn
    layers = []
    if separable and (ks > 1):
        layers += [conv3d(ni, ni, ks=ks, stride=stride, padding=padding, bias=bias, norm_type=norm_type, init=init, groups=ni)]
        layers += [conv3d(ni, nf, ks=1, stride=1, padding=0, bias=bias, norm_type=norm_type, init=init)]
    else:
        layers += [conv3d(ni, nf, ks=ks, stride=stride, padding=padding, bias=bias, norm_type=norm_type, init=init)]
    
    if use_activ: layers.append(relu(True, leaky=leaky))
    if bn: layers.append(nn.BatchNorm3d(nf))
    if self_attention: layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)

def res_block3d(nf, dense:bool=False, norm_type:Optional[NormType]=NormType.Batch, bottle:bool=False, **kwargs):
    "Resnet block of `nf` features."
    norm2 = norm_type
    if not dense and (norm_type==NormType.Batch): norm2 = NormType.BatchZero
    nf_inner = nf//2 if bottle else nf
    return SequentialEx(conv_layer3d(nf, nf_inner, norm_type=norm_type, **kwargs),
                      conv_layer3d(nf_inner, nf, norm_type=norm2, **kwargs),
                      MergeLayer(dense))

class ResLayer3d(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.conv1 = conv_layer3d(ni, ni//2, ks=1)
        self.conv2 = conv_layer3d(ni//2, ni, ks=3)
    def forward(self, x): return x + self.conv2(self.conv1(x))

class Apply2dConv(nn.Module):
    def __init__(self, arch, pool=True):
        super().__init__()
        layers = [arch]
        if pool: layers += [nn.AdaptiveMaxPool2d(1), Flatten()]
        self.arch = nn.Sequential(*layers)
        
    def forward(self, x):
        b, t, *size = x.shape
        x = x[:,:,None].expand([b,t,3,*size])
        out = [self.arch(x[:,i])[:,None] for i in range(t)]
        return torch.cat(out, dim=1)

class ReshapeSeg2d(nn.Module):
    '''Reshape outputs from 2d segmentation models'''
    def forward(self, x):
        _,*c = x.shape
        return x.view(-1,25,*c)