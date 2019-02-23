from fastai.basics import *
from .layers import *
from .utils import classifier_report, try_sm
from types import MethodType
from sklearn.metrics import roc_auc_score

__all__ = ['LossMetrics', 'AucLogger', 'ParallelAucLogger']

class LossMetrics(callbacks.LossMetrics):
    'Modification for Parallel models'
    def on_batch_end(self, last_target, train, **kwargs):
        "Update the metrics if not `train`"
        if train: return
        bs = last_target[0].size(0)
        for name in self.names:
            self.metrics[name] += bs * self.learn.loss_func.metrics[name].detach().cpu()
        self.nums += bs

class AucLogger(LearnerCallback):
    _order=-19
    def reset_history(self):
        self.y_true = []
        self.y_score = []
        self.y_score_train = None
        self.y_true_train = None
        
    def calc_auc(self):
        self.get_validation_preds()
        return roc_auc_score(self.y_true, try_sm(self.y_score)[:,1])

    def calc_train_auc(self):
        self.get_train_preds()
        return roc_auc_score(self.y_true_train, try_sm(self.y_score_train)[:,1])

    def get_train_preds(self):
        loss_func = self.learn.loss_func
        self.learn.loss_func = None
        if self.y_score_train is None:
            self.y_score_train, self.y_true_train = self.learn.get_preds(DatasetType.Train)
        self.learn.loss_func = loss_func
        return self.y_score_train, self.y_true_train

    def get_validation_preds(self):
        loss_func = self.learn.loss_func
        self.learn.loss_func = None
        if len(self.y_score) == 0:
            self.y_score, self.y_true = self.learn.get_preds(DatasetType.Valid)
        self.learn.loss_func = loss_func
        return self.y_score, self.y_true
    
    def print_report(self, ds_type=None):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            if   ds_type == DatasetType.Train: classifier_report(*self.get_train_preds(), title='Train Set')
            elif ds_type == DatasetType.Valid: classifier_report(*self.get_validation_preds(), title='Validation Set')
            else:
                axs1,lims1 = classifier_report(*self.get_train_preds(), title='Train Set')
                axs2,lims2 = classifier_report(*self.get_validation_preds(), title='Validation Set')
                lims = min(lims1+lims2),max(lims1+lims2)
                axs1[-1].set_xlim(lims)
                axs2[-1].set_xlim(lims)
        
    def on_batch_end(self, train, last_output, last_target, **kwargs):
        if not train:
            self.y_true  += [last_target.detach().cpu()]
            self.y_score += [last_output.detach().cpu()]
    
    def on_epoch_begin(self, **kwargs):
        self.reset_history()

    def on_epoch_end(self, train, **kwargs):
        if (not train) and (len(self.y_true) > 0):
            self.y_true  = torch.cat(self.y_true)
            self.y_score = torch.cat(self.y_score)
            auc = self.calc_auc()
            if len(self.recorder._added_met_names) == 1: # There are no other metrics
                self.recorder.add_metrics([auc])
            else:
                metrics = self.recorder._added_mets if hasattr(self.recorder, '_added_mets') else []
                self.recorder.add_metrics(metrics + [auc])
    
    def on_train_begin(self, **kwargs):
        met_names = self.recorder._added_met_names if hasattr(self.recorder, '_added_met_names') else []
        self.recorder.add_metric_names(met_names + ['auc'])
        self.reset_history()

class ParallelAucLogger(AucLogger):
    def __post_init__(self): 
        super().__post_init__()
        self.learn.get_preds = MethodType(parallel_get_preds, self.learn)

    def get_train_preds(self):
        loss_func = self.learn.loss_func
        self.learn.loss_func = None
        if self.y_score_train is None:
            _, self.y_score_train, _, self.y_true_train = self.learn.get_preds(DatasetType.Train, get_segs=False)
        self.learn.loss_func = loss_func
        return self.y_score_train, self.y_true_train

    def get_validation_preds(self):
        loss_func = self.learn.loss_func
        self.learn.loss_func = None
        if len(self.y_score) == 0:
            _, self.y_score, _, self.y_true = self.learn.get_preds(DatasetType.Valid, get_segs=False)
        self.learn.loss_func = loss_func
        return self.y_score, self.y_true
    
    def on_batch_end(self, train, last_output, last_target, **kwargs):
        if not train:
            self.y_true  += [last_target[1].detach().cpu()]
            self.y_score += [last_output[1].detach().cpu()]    

def parallel_loss_batch(model:nn.Module, xb:Tensor, yb:Tensor, loss_func:OptLossFunc=None, opt:OptOptimizer=None,
                      cb_handler:Optional[CallbackHandler]=None)->Tuple[Union[Tensor,int,float,str]]:
    "Calculate loss and metrics for a batch, call out to callbacks as necessary."
    cb_handler = ifnone(cb_handler, CallbackHandler())
    if not is_listy(xb): xb = [xb]
    if not is_listy(yb): yb = [yb]
    out = model(*xb)
    out = cb_handler.on_loss_begin(out)

    if not loss_func: return to_detach(out), (yb[0].detach(), yb[1].detach())
    loss = loss_func(out, *yb)

    if opt is not None:
        loss = cb_handler.on_backward_begin(loss)
        loss.backward()
        cb_handler.on_backward_end()
        opt.step()
        cb_handler.on_step_end()
        opt.zero_grad()

    return loss.detach().cpu()

def parallel_validate(model:nn.Module, dl:DataLoader, loss_func:OptLossFunc=None, cb_handler:Optional[CallbackHandler]=None,
                    pbar:Optional[PBar]=None, average=True, n_batch:Optional[int]=None)->Iterator[Tuple[Union[Tensor,int],...]]:
    "Calculate loss and metrics for the validation set."
    model.eval()
    with torch.no_grad():
        val_losses,nums = [],[]
        for xb,yb in progress_bar(dl, parent=pbar, leave=(pbar is not None)):
            if cb_handler: xb, yb = cb_handler.on_batch_begin(xb, yb, train=False)
            val_losses.append(parallel_loss_batch(model, xb, yb, loss_func, cb_handler=cb_handler))
            if not is_listy(yb): yb = [yb]
            nums.append(yb[0].shape[0])
            if cb_handler and cb_handler.on_batch_end(val_losses[-1]): break
            if n_batch and (len(nums)>=n_batch): break
        nums = np.array(nums, dtype=np.float32)
        if average: return (to_np(torch.stack(val_losses)) * nums).sum() / nums.sum()
        else:       return val_losses

def parallel_get_preds(self, ds_type=DatasetType.Valid, get_segs=False):
    reshape_seg = ReshapeSeg2d()
    x1,x2,y1,y2 = [],[],[],[]
    for (tx1,tx2),(ty1,ty2) in parallel_validate(self.model, self.dl(ds_type), average=False):
        x2 += [tx2]
        y2 += [ty2]
        if get_segs:
            x1 += [reshape_seg(tx1)]
            y1 += [ty1]

    x2 = torch.cat(x2).cpu()
    y2 = torch.cat(y2).cpu()
    if get_segs:
        x1 = torch.cat(x1).cpu()
        y1 = torch.cat(y1).cpu()

    return x1,x2,y1,y2