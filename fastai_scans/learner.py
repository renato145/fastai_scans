from fastai.basics import *
from .utils import classifier_report, regression_report, plot_cm

__all__ = ['ClassificationInterpretation', 'RegressionInterpretation']

class ClassificationInterpretation():
    def __init__(self, learn:Learner):
        self.learn = learn
        self.data = {}
    @property
    def model(self): return self.learn.model
    @property
    def reconstruct(self): return self.learn.data.x.reconstruct
    @property
    def loss_func(self): return self.learn.loss_func
    @loss_func.setter
    def loss_func(self, loss_func): self.learn.loss_func = loss_func
    @property
    def classes(self): return self.learn.data.y.classes

    def refresh(self, ds_type=DatasetType.Valid):
        loss_func = self.loss_func
        self.loss_func = None
        y_score,y_true = self.learn.get_preds(ds_type=ds_type)
        self.loss_func = loss_func
        return y_score,y_true

    def get_data(self, ds_type):
        if self.data.get(ds_type) is None: self.data[ds_type] = self.refresh(ds_type)
        return self.data[ds_type]

    def show_report(self, ds_type=None, figsize=(12,5)):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            if ds_type is None:
                axs1,lims1 = classifier_report(*self.get_data(DatasetType.Train), figsize=figsize, title='Train Set', labels=self.classes)
                axs2,lims2 = classifier_report(*self.get_data(DatasetType.Valid), figsize=figsize, title='Validation Set', labels=self.classes)
                lims = min(lims1+lims2),max(lims1+lims2)
                axs1[-1].set_xlim(lims)
                axs2[-1].set_xlim(lims)
            else:
                classifier_report(*self.get_data(ds_type), figsize=figsize, title=f'{ds_type._name_} Set', labels=self.classes)

class RegressionInterpretation(ClassificationInterpretation):
    def __init__(self, learn:Learner, scale:float=1.0):
        super().__init__(learn)
        self.scale = scale

    @property
    def classes(self): return 1
    def refresh(self, ds_type=DatasetType.Valid): return [e*self.scale for e in super().refresh(ds_type=ds_type)]

    def show_report(self, ds_type=None, samples=50, figsize=(12,4)):
        if ds_type is None:
            regression_report(*self.get_data(DatasetType.Train), figsize=figsize, title='Train Set', samples=samples)
            regression_report(*self.get_data(DatasetType.Valid), figsize=figsize, title='Validation Set', samples=samples)
        else:
            regression_report(*self.get_data(ds_type), figsize=figsize, title=f'{ds_type._name_} Set', samples=samples)

    def brats_classification(self, ds_type=None, figsize=(4,4)):
        if ds_type == None:
            fig,axs = plt.subplots(1, 2, figsize=(figsize[0]*2,figsize[1]))
            brats_classification(*self.get_data(DatasetType.Train), ax=axs[0], figsize=figsize, title='Train Set')
            brats_classification(*self.get_data(DatasetType.Valid), ax=axs[1], figsize=figsize, title='Validation Set')
            plt.tight_layout()
        else:
            brats_classification(*self.get_data(ds_type), figsize=figsize, title=f'{ds_type._name_} Set')

def brats_classification(y_score, y_true, ax=None, figsize=(4,4), title=None):
    labels = ['short', 'mid', 'long']
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    y = pd.cut(Series(y_true.flatten()), [0, 10*30, 15*30, np.inf], labels=labels)
    y_ = pd.cut(Series(y_score.flatten()), [0, 10*30, 15*30, np.inf], labels=labels)
    ax,acc,_ = plot_cm(y_, y, ax=ax, figsize=figsize, labels=labels)
    t = '' if title is None else r'$\bf{' + title + '}$\n'
    t += f'Accuracy={acc:.4f}'
    ax.set_title(t, size=14)
