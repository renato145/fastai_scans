from fastai.basics import *
from fastai.callbacks.hooks import *
from ..layers import *
from .segmentation import *
from .segmentation import accuracy as seg_accuracy, dice as _dice
from .classification import ClassifierHead

__all__ = ['get_parallel_metrics', 'ParallelModel']

def cls_acc(input, targ_seg, targ_lbl):
    seg, lbl = input
    return accuracy(lbl, targ_lbl)

def seg_acc(input, targ_seg, targ_lbl):
    seg, lbl = input
    return seg_accuracy(seg, targ_seg)

def seg_dice(input, targ_seg, targ_lbl):
    seg, lbl = input
    return _dice(seg, targ_seg)

def get_parallel_metrics(): return [seg_dice, seg_acc, cls_acc]

def get_first_vnetblock_idx(m):
    for i,l in enumerate(m):
        if isinstance(l, VnetBlock): return i
    else: raise Exception('No VnetBlocks in unet module.')

class ParallelModel(nn.Module):
    def __init__(self, vol_size, unet, hidden=200, num_classes=2, drop_conv=0, drop_out=0,
                 concat_pool=False, detach=True):
        super().__init__()
        self.unet = unet
        # Grab encoder result
        i = get_first_vnetblock_idx(unet.layers) - 1
        self.hook = hook_output(unet.layers[i], detach=detach)
        # Classifier
        nf = flatten_model(unet.layers[i])[-1].weight.shape[0]
        if concat_pool:
            pool = AdaptiveConcatPool3d()
            nf *= 2
        else:
            pool = nn.AdaptiveMaxPool3d(1)
        self.classifier = nn.Sequential(pool, Flatten(),
                                        ClassifierHead([nf,hidden,num_classes],[drop_conv,drop_out]))
        
    def forward(self, x): return self.unet(x), self.classifier(self.hook.stored)
