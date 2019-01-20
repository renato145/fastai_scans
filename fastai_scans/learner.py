from fastai.basics import *
from .models import *
from .layers import *
from .callbacks import *
from fastai.callbacks import CSVLogger

__all__ = ['ScansLearner', 'get_pretrained_learner', 'get_2dpretrained_learner', 'get_2dpretrained_conv1_learner', 'get_2dpretrained_lstm_learner']

def pretrain_split(m): return (m[1],)

def get_output_shape(m,x):
    with torch.no_grad():
        s = m.eval()(x)
    return s.shape

class ScansLearner(Learner):
    def __init__(self, data, model, initialization=nn.init.kaiming_normal_, **kwargs):
        self.initialization = initialization
        super().__init__(data, model, **kwargs)

    def __post_init__(self):
        super().__post_init__()
        if self.initialization is not None: self.init(nn.init.kaiming_normal_)

def get_pretrained_learner(path, data, encoder, head, loss_func=F.cross_entropy,
                           metrics=accuracy, callback_fns=[AucLogger, CSVLogger], device=defaults.device):
    m = nn.Sequential(encoder, head)
    learn = ScansLearner(data, m, loss_func=loss_func, metrics=metrics, path=path, callback_fns=callback_fns, initialization=None)
    learn.split(pretrain_split)
    apply_init(m[1], nn.init.kaiming_normal_)
    learn.freeze()

    return learn

def get_2dpretrained_learner(path, data, arch, head_layers, head_drops, loss_func=F.cross_entropy, metrics=accuracy,
                             callback_fns=[AucLogger, CSVLogger], device=defaults.device):
    body = Apply2dConv(arch).to(device)
    x, y = data.one_batch()
    b,t,c = get_output_shape(body, x.to(device))
    nf = [t*c]
    head = nn.Sequential(Flatten(), ClassifierHead(nf+head_layers, head_drops))
    return get_pretrained_learner(path, data, body, head, loss_func, metrics, callback_fns, device)

def get_2dpretrained_conv1_learner(path, data, arch, conv_layers, head_layers, head_drops, loss_func=F.cross_entropy, metrics=accuracy,
                                  callback_fns=[AucLogger, CSVLogger], device=defaults.device):
    body = Apply2dConv(arch).to(device)
    x, y = data.one_batch()
    nf = get_output_shape(body, x.to(device))[1]
    head = Conv1dClassifierHead(nf, conv_layers, head_layers, head_drops)
    return get_pretrained_learner(path, data, body, head, loss_func, metrics, callback_fns, device)

def get_2dpretrained_lstm_learner(path, data, arch, nf_rnn, head_layers, head_drops, bidirectional=False, pool=False,
                                  loss_func=F.cross_entropy, metrics=accuracy, callback_fns=[AucLogger, CSVLogger], device=defaults.device):
    body = Apply2dConv(arch).to(device)
    x, y = data.one_batch()
    nf = get_output_shape(body, x.to(device))[2]
    head = LstmClassifierHead(nf, nf_rnn, head_layers, head_drops, bidirectional=bidirectional, pool=pool).cuda()
    return get_pretrained_learner(path, data, body, head, loss_func, metrics, callback_fns, device)
