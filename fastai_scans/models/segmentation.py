from fastai.basics import *
from fastai.callbacks.hooks import *
from ..layers import *

__all__ = ['get_segmentation_metrics', 'VnetBlock', 'DynamicVnet']

def accuracy(input, targs):
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    return (input==targs).float().mean()

def _dice(a, b, iou=False):
    intersect = (a * b).sum().float()
    union = (a + b).sum().float()
    if not iou: return (2. * intersect / union if union > 0 else union.new([1.]).squeeze())
    else: return intersect / (union-intersect+1.0)
    
def get_dice_coefs(input, targs):
    n, c, *_ = input.shape
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    dice_coefs = [_dice(input==e, targs==e) for e in range(c)]
    return dice_coefs

def dice(input, targs):
    dice_coefs = get_dice_coefs(input, targs)
    return sum(dice_coefs)/input.shape[1]

def get_segmentation_metrics(): return [accuracy, dice]

def _get_sfs_idxs(sizes:Sizes) -> List[int]:
    "Get the indexes of the layers where the size of the activation changes."
    feature_szs = [size[-3:] for size in sizes]
    sfs_idxs = list(np.where(np.any(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]), axis=1))[0])
    if feature_szs[0] != feature_szs[1]: sfs_idxs = [0] + sfs_idxs
    return sfs_idxs

class VnetBlock(nn.Module):
    def __init__(self, up_in_c:int, x_in_c:int, hook:Hook, final_div:bool=True, do_shuffle:bool=False,
                 double_out_conv=False, leaky:float=None, self_attention:bool=False, **kwargs):
        super().__init__()
        self.hook = hook
        
        if do_shuffle:
            self.upsample = PixelShuffle3d_ICNR(up_in_c, up_in_c//2, leaky=leaky, **kwargs)
        else:
            self.upsample = nn.Sequential(Upsample3dLike(hook.stored),
                                          conv_layer3d(up_in_c, up_in_c//2, ks=1))
        
        self.bn = batchnorm_3d(x_in_c)
        ni = up_in_c//2 + x_in_c
        nf = ni if final_div else ni//2
        convs =  [conv_layer3d(ni, nf, leaky=leaky, **kwargs)] if double_out_conv else []
        convs += [conv_layer3d(nf if double_out_conv else ni, nf, leaky=leaky, self_attention=self_attention, **kwargs)]
        self.convs = nn.Sequential(*convs)
        self.relu = relu(leaky=leaky)
        
    def forward(self, up_in):
        s = self.hook.stored
        up_out = self.upsample(up_in)
        ssh = s.shape[-3:]
        if ssh != up_out.shape[-3:]:
            up_out = F.interpolate(up_out, s.shape[-3:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.convs(cat_x)

class DynamicVnet(SequentialEx):
    "Create a V-Net from a given architecture."
    def __init__(self, vol_size, encoder:nn.Module, n_classes:int, last_cross:bool=True, bottle:bool=False,
                 pixel_shuffle=False, light_up_block=True, self_attention:bool=False, **kwargs):
        sfs_szs = model_sizes(encoder, size=vol_size)
        # sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        sfs_idxs = np.unique(_get_sfs_idxs(sfs_szs))[::-1].tolist()
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])
        x = dummy_eval(encoder, vol_size).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(conv_layer3d(ni, ni*2, **kwargs),
                                    conv_layer3d(ni*2, ni, **kwargs)).eval()
        x = middle_conv(x)
        layers = [encoder, batchnorm_3d(ni), nn.ReLU(), middle_conv]

        for i,idx in enumerate(sfs_idxs):
            not_final = i!=len(sfs_idxs)-1
            ds = not any([a>=b for a,b in zip(x.shape[-3:], sfs_szs[idx][-3:])])
            ds = ds if pixel_shuffle else pixel_shuffle
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            sa = self_attention and (i==len(sfs_idxs)-3)
            vnet_block = VnetBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final, do_shuffle=ds,
                                   double_out_conv=not light_up_block, self_attention=sa, **kwargs).eval()
            layers.append(vnet_block)
            x = vnet_block(x)

        ni = x.shape[1]
        if vol_size != sfs_szs[0][-3:]:
            #layers.append(PixelShuffle_ICNR(ni, **kwargs))
            layers.append(Upsample3d(vol_size))
            
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(res_block3d(ni, bottle=bottle, **kwargs))
            
        layers += [conv_layer3d(ni, n_classes, ks=1, use_activ=False, **kwargs)]
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()
