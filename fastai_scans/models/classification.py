from fastai.basics import *
from fastai.callbacks.hooks import dummy_eval
from ..layers import *

__all__ = ['Simple3d', 'ResNet3d', 'DenseNet3d', 'ClassifierHead', 'Conv1dClassifierHead', 'LstmClassifierHead']

def conv_pool(ni, nf, ks=3, stride=1, do_pooling=True, pool_func=nn.MaxPool3d, **kwargs):
    pool = do_pooling and any([e>1 for e in listify(stride)])
    conv = conv_layer3d(ni, nf, ks=ks, stride=1 if pool else stride, **kwargs)
    if pool: return nn.Sequential(conv, pool_func(stride))
    else   : return conv
    
class Simple3d(nn.Sequential):
    def __init__(self, vol_size, num_layers, ni=1, nf=16, hidden=200, num_classes=2, drop_conv=0, drop_out=0,
                 separable_convs=False, concat_pool=False, self_attention=False, do_pooling=True):
        strides = [2 if e > max(vol_size)//2 else 1 for e in vol_size]
        layers = [conv_pool(ni, nf, stride=strides, do_pooling=do_pooling, separable=separable_convs)]
        x = dummy_eval(layers[0], vol_size).detach()
        
        for i in range(num_layers-1):
            dims = x.shape[-3:]
            strides = 1 if i==num_layers-2 else [2 if e > max(dims)//2 else 1 for e in dims]
            sa = self_attention and (i==num_layers-4)
            layers.append(conv_pool(nf, nf*2, stride=strides, do_pooling=do_pooling, separable=separable_convs, self_attention=sa))
            nf *= 2
            x = layers[-1].eval()(x)
        
        if concat_pool:
            pool = AdaptiveConcatPool3d()
            nf *= 2
        else:
            pool = nn.AdaptiveMaxPool3d(1)
        
        layers += [pool, Flatten(), ClassifierHead([nf,hidden,num_classes],[drop_conv,drop_out])]
        super().__init__(*layers)

class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        downsample = any([e>1 for e in listify(stride)])
        self.convs = nn.Sequential(conv_layer3d(ni, nf, stride=stride),
                                   conv_layer3d(nf, nf, use_activ=False))
        self.downsample = (conv_layer3d(ni, nf, ks=1, stride=stride, use_activ=False)
                           if downsample or (ni != nf) else None)
        self.activ = relu(True)
        

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        return self.activ(self.convs(x) + res)

class ResNet3d(nn.Sequential):
    def make_group_layer(self, ch_in, num_blocks, stride):
        return nn.Sequential(ResBlock(ch_in,ch_in*2, stride=stride),
                             *[ResBlock(ch_in*2,ch_in*2) for i in range(num_blocks)])

    def __init__(self, vol_size, num_blocks=[2,2,2,2], ni=1, nf=16, num_classes=2, drop_out=0,
                 concat_pool=False, final_pool=nn.AdaptiveAvgPool3d):
        layers = [conv_pool(ni, nf, ks=7, stride=2, do_pooling=False),
                  nn.MaxPool3d(kernel_size=3, stride=2, padding=1)]
        x = dummy_eval(layers[0], vol_size).detach()
        x = layers[-1].eval()(x)
        
        for i,nb in enumerate(num_blocks):
            dims = x.shape[-3:]
            final = i == len(num_blocks)-1
            strides = 1 if (final or i==0) else [2 if e > max(dims)//2 else 1 for e in dims]
            layers.append(self.make_group_layer(nf, nb, stride=strides))
            if not final: x = layers[-1].eval()(x)
            nf *= 2
            
        if concat_pool:
            pool = AdaptiveConcatPool3d()
            nf *= 2
        else:
            pool = final_pool(1)
        
        layers += [pool, Flatten(), ClassifierHead([nf,num_classes],[drop_out])]
        super().__init__(*layers)

class DenseBlock(SequentialEx):
    def __init__(self, ni, growth_rate, bn_size, drop_rate):
        layers = [conv_layer3d(ni, bn_size*growth_rate, ks=1),
                  conv_layer3d(bn_size*growth_rate, growth_rate)]
        if drop_rate != 0: layers.append(nn.Dropout(drop_rate))
        layers.append(MergeLayer(dense=True))
        super().__init__(*layers)
    
class DenseNet3d(nn.Sequential):
    def make_dense_layer(self, ch_in, num_blocks, stride, bn_size, growth_rate, drop_rate, transition):
        layers = []
        for i in range(num_blocks):
            layers.append(DenseBlock(ch_in + i*growth_rate, growth_rate, bn_size, drop_rate))
        ni = ch_in + num_blocks*growth_rate
        nf = ni//2
        if transition: layers.append(conv_pool(ni, nf, ks=1, stride=stride, pool_func=nn.AvgPool3d))
        return nn.Sequential(*layers)

    def __init__(self, vol_size, growth_rate=32, num_blocks=[2,2,2,2], ni=1, nf=16, bn_size=4, num_classes=2,
                 drop_rate=0, drop_out=0, concat_pool=False, final_pool=nn.AdaptiveAvgPool3d):
        layers = [conv_pool(ni, nf, ks=7, stride=2, do_pooling=False),
                  nn.MaxPool3d(kernel_size=3, stride=2, padding=1)]
        x = dummy_eval(layers[0], vol_size).detach()
        x = layers[-1].eval()(x)
        
        for i,nb in enumerate(num_blocks):
            dims = x.shape[-3:]
            final = i == len(num_blocks)-1
            strides = 1 if final else [2 if e > max(dims)//2 else 1 for e in dims]
            layers.append(self.make_dense_layer(nf, nb, stride=strides, bn_size=bn_size, growth_rate=growth_rate,
                                                drop_rate=drop_rate, transition=not final))
            nf += nb*growth_rate
            if not final:
                nf //= 2
                x = layers[-1][-1][-1].eval()(x)
            
        if concat_pool:
            pool = AdaptiveConcatPool3d()
            nf *= 2
        else:
            pool = final_pool(1)
        
        layers += [pool, Flatten(), ClassifierHead([nf,num_classes], [drop_out])]
        super().__init__(*layers)

class ClassifierHead(nn.Sequential):
    def __init__(self, layers, drops):
        if not is_listy(drops): drops = listify(drops, layers[:-1])
        layers = [l for l in layers if l is not None]
        drops  = [l for l in drops  if l is not None]
        mod_layers = []
        activs = [nn.ReLU(inplace=True)] * (len(layers) - 2) + [None if layers[-1]>1 else nn.Sigmoid()]
        for n_in,n_out,p,actn in zip(layers[:-1],layers[1:], drops, activs):
                mod_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        super().__init__(*mod_layers)

class Conv1dClassifierHead(nn.Sequential):
    def __init__(self, features_in, conv_layers, layers, drops, ks=5, stride=3):
        mod_layers = []
        for i, (n_in,n_out) in enumerate(zip([features_in]+conv_layers[:-1], conv_layers)):
            mod_layers += [nn.BatchNorm1d(n_in),
                           nn.Conv1d(n_in, n_out, kernel_size=ks, stride=stride), nn.ReLU(inplace=True)]
            if i == len(conv_layers)-1: mod_layers += [nn.AdaptiveMaxPool1d(1), Flatten()]
        
        mod_layers += [ClassifierHead([n_out]+layers, drops)]
        super().__init__(*mod_layers)
        
class LstmClassifierHead(nn.Module):
    def __init__(self, features_in, nf_rnn, layers, drops, bidirectional=False, pool=False):
        super().__init__()
        self.ndir = 2 if bidirectional else 1
        self.rnn = nn.LSTM(features_in, nf_rnn, 1, bidirectional=bidirectional).cuda()
        self.do_pool = pool
        nf = nf_rnn
        if bidirectional: nf *= 2
        if pool: nf *= 3
        self.layers = ClassifierHead([nf]+layers, drops)
        
    def pool(self, x, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1,2,0), (1,)).squeeze()
        
    def forward(self, x):
        out, *_ = self.rnn(x.transpose(0,1))
        if self.do_pool:
            avgpool = self.pool(out, False)
            mxpool = self.pool(out, True)
            out = torch.cat([out[-1], mxpool, avgpool], 1)
        else:
            out = out[-1]
        
        return self.layers(out)
