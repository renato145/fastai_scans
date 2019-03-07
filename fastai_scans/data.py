import bcolz
from fastai.basics import *
from .volume import *
from .layers import *

__all__ = ['BcolzDataBunch', 'VolumeItemList', 'SegmentationLabelList', 'SegmentationItemList', 'ParallelLabelList', 'ParallelItemList']

def self_normalize(x, **kwargs):
    mean = x.mean([2,3,4])[...,None,None,None]
    std = x.view(*x.shape[:2], -1).std(-1)[...,None,None,None]
    return (x-mean)/std

def normalize_fun(b, do_x=True, do_y=False):
    x,y = b
    if do_x: x = self_normalize(x)
    if do_y: y = self_normalize(y)
    return x,y

def denormalize(x): return x.cpu()

class BcolzDataBunch(DataBunch):
    @classmethod
    def create(cls, train_ds, valid_ds, **kwargs):
        if 'num_workers' in kwargs: kwargs.pop('num_workers')
        return super().create(train_ds, valid_ds, num_workers=0, **kwargs)

    def normalize(self, do_x=True, do_y=False):
        if getattr(self,'norm',False): raise Exception('Can not call normalize twice')
        self.norm = partial(normalize_fun, do_x=do_x, do_y=do_y)
        self.denorm = denormalize
        self.add_tfm(self.norm)
        return self

class VolumeItemList(ItemList):
    _item_cls,_bunch,_square_show  = Volume,BcolzDataBunch,False
    def __init__(self, items, bcolz_array:bcolz.carray, metadata:pd.DataFrame, tfm_params=None, **kwargs):
        self.bcolz_array = bcolz_array
        self.metadata = metadata
        self.tfm_params = tfm_params
        super().__init__(items, **kwargs)
    
    def new(self, items, **kwargs):
        return super().new(items, bcolz_array=self.bcolz_array, metadata=self.metadata, tfm_params=self.tfm_params, **kwargs)
    
    def get(self, i):
        idx = super().get(i)
        metadata = None if self.metadata is None else self.metadata.iloc[idx]
        tfm_params = None if self.tfm_params is None else self.tfm_params[idx]
        return self._item_cls(self.bcolz_array[idx], idx=idx, metadata=metadata, tfm_params=tfm_params)
    
    @classmethod
    def from_carray(cls, bcolz_array, metadata, tfm_params=None, items=None, **kwargs):
        if items is None: items = range(len(bcolz_array))
        return cls(items, bcolz_array, metadata, tfm_params=tfm_params, **kwargs)

    @classmethod
    def from_paths(cls, path, metadata_path, tfm_params=None, **kwargs):
        '''
        Use ex:
        data = (fastai_scans.VolumeItemList.from_paths(path / 'train_scans128', path/'train_metadata.csv')
                                           .split_by_rand_pct(0.2, seed=7)
                                           .label_from_metadata('mort')
                                           .transform(fastai_scans.get_transforms())
                                           .databunch(bs=8)
                                           .normalize())
        '''
        return cls.from_carray(bcolz.open(path, mode='r'), pd.read_csv(metadata_path), tfm_params, **kwargs)

    def filter_by_idxs(self, idxs):
        self.items = idxs
        return self

    def filter_na_lbls(self, col_name):
        col = self.metadata[col_name]
        if col.isna().sum() > 0:
            tmp_lbl = col.dropna().iloc[0]
            idxs = self.metadata.index[col.notna()].to_numpy()
            col.fillna(tmp_lbl, inplace=True)
            return self.filter_by_idxs(idxs)
        else:
            return self

    def split_by_metadata(self, col_name):
        valid_idx = np.where(self.metadata[col_name]==1)[0]
        return super().split_by_idx(valid_idx)

    def label_from_metadata(self, col_name, **kwargs):
        return self._label_from_list(self.metadata.iloc[self.items][col_name], **kwargs)

    def reconstruct(self, t:Tensor): return self._item_cls(t)

    def show_xys(self, xs, ys, channel=0, n_slices=4, imgsize=3, figsize=None, **kwargs):
        rows = len(xs)
        axs = subplots(rows, n_slices, imgsize=imgsize, figsize=figsize)
        for i, row_axs in enumerate(axs):
            xs[i].show(axes=row_axs, label=ys[i], channel=channel, **kwargs)
        plt.tight_layout()
        
    def show_xyzs(self, xs, ys, zs, channel=0, n_slices=4, imgsize=3, figsize=None, y_scale=None, **kwargs):
        rows = len(xs)
        axs = subplots(rows, n_slices, imgsize=imgsize, figsize=figsize)
        for i, row_axs in enumerate(axs):
            if isinstance(zs[i].data, (tuple,list,np.ndarray)):
                pred = zs[i].data[0] if len(zs[i].data)==1 else zs[i].data
            else: pred = zs[i]
            if y_scale is not None: pred *= y_scale
            xs[i].show(axes=row_axs, label=ys[i], extra_lbl=['Prediction'], extra_inf=[pred], channel=channel,
                       y_scale=y_scale, **kwargs)
        plt.tight_layout()

class SegmentationLabelList(VolumeItemList):
    _item_cls = VolumeSegment
    def __init__(self, items, bcolz_array:bcolz.carray, metadata:pd.DataFrame, tfm_params=None, **kwargs):
        super().__init__(items, bcolz_array, metadata, tfm_params, **kwargs)
        self.loss_func = SegCrossEntropy()

    def analyze_pred(self, pred, thresh:float=0.5): return pred.argmax(0)

class SegmentationItemList(VolumeItemList):
    _label_cls = SegmentationLabelList
    def __init__(self, items, bcolz_array:bcolz.carray, bcolz_array_masks:bcolz.carray,
                 metadata:pd.DataFrame, tfm_params:np.array=None, **kwargs):
        super().__init__(items, bcolz_array, metadata, tfm_params, **kwargs)
        self.bcolz_array_masks = bcolz_array_masks
    
    def new(self, items, **kwargs):
        return super(VolumeItemList, self).new(items, bcolz_array=self.bcolz_array, bcolz_array_masks=self.bcolz_array_masks,
                                               metadata=self.metadata, tfm_params=self.tfm_params, **kwargs)
    
    @classmethod
    def from_carray(cls, bcolz_array, bcolz_array_masks, metadata=None, tfm_params=None):
        items = range(len(bcolz_array))
        return cls(items, bcolz_array, bcolz_array_masks, metadata, tfm_params)

    @classmethod
    def from_paths(cls, path, path_masks, metadata_path=None, tfm_params=None):
        '''
        Use ex:
        data = (fastai_scans.SegmentationItemList.from_paths(path/'train_scans128', path/'train_segmentation', path/'train_metadata.csv')
                                                 .split_by_rand_pct(0.2, seed=7)
                                                 .label_from_bcolz()
                                                 .transform(fastai_scans.get_transforms(), tfm_y=True)
                                                 .databunch(bs=4)
                                                 .normalize())
        '''
        metadata = metadata_path if metadata_path is None else pd.read_csv(metadata_path)
        return cls.from_carray(bcolz.open(path, mode='r'), bcolz.open(path_masks, mode='r'), metadata, tfm_params)
    
    def label_from_bcolz(self, from_item_lists=False, **kwargs):
        y = self._label_cls.from_carray(self.bcolz_array_masks, self.metadata, self.tfm_params,
                                        items=self.items, path=self.path, **kwargs)
        res = self._label_list(x=self, y=y)
        return res
    
    def show_xys(self, xs, ys, channel=0, n_slices=4, imgsize=4, figsize=None,
                 x_cmap='magma', y_cmap='magma', slice_info=False, **kwargs):
        rows = len(xs)
        axs = seg_subplots(rows, n_slices, n_segs=2, imgsize=imgsize, figsize=figsize)
        for i, row_axs in enumerate(axs):
            xs[i].show(axes=row_axs[::2], label=None, cmap=x_cmap, channel=channel, **kwargs)
            ys[i].show(axes=row_axs[1::2], label=None, cmap=y_cmap, title='labels', slice_info=slice_info, channel=channel, **kwargs)
        plt.tight_layout()
        
    def show_xyzs(self, xs, ys, zs, channel=0, n_slices=3, imgsize=4, figsize=None,
                  x_cmap='magma', y_cmap='magma', slice_info=False, **kwargs):
        rows = len(xs)
        axs = seg_subplots(rows, n_slices, n_segs=3, imgsize=imgsize, figsize=figsize)
        for i, row_axs in enumerate(axs):
            xs[i].show(axes=row_axs[::3], label=None, cmap=x_cmap, channel=channel, **kwargs)
            ys[i].show(axes=row_axs[1::3], label=None, cmap=y_cmap, title='labels', slice_info=slice_info, channel=channel, **kwargs)
            zs[i].show(axes=row_axs[2::3], label=None, cmap=y_cmap, title='preds', slice_info=slice_info, channel=channel, **kwargs)
        plt.tight_layout()

class VolumeSegmentLblProcessor(CategoryProcessor):
    def process(self, ds):
        if self.classes is None: self.create_classes(self.generate_classes(ds.labels))
        ds.classes = self.classes
        ds.c2i = self.c2i
        ds.labels = array([self.process_one(item) for item in ds.labels])

class ParallelLabelList(VolumeItemList):
    _item_cls,_processor = VolumeSegmentLbl,VolumeSegmentLblProcessor
    def __init__(self, items, bcolz_array:bcolz.carray, labels, metadata:pd.DataFrame, tfm_params=None,
                 classes=None, **kwargs):
        super().__init__(items, bcolz_array, metadata, tfm_params, **kwargs)
        self.labels,self.classes,self.loss_func = labels,classes,ParallelLoss(regression=False)
    
    def new(self, items, **kwargs):
        return super(VolumeItemList, self).new(items, bcolz_array=self.bcolz_array, labels=self.labels, metadata=self.metadata,
                                               tfm_params=self.tfm_params, classes=self.classes, **kwargs)
    
    def get(self, i):
        idx = self.items[i]
        return self._item_cls(self.bcolz_array[idx], self.labels[idx], self.classes[self.labels[idx]],
                              idx=idx, metadata=self.metadata.iloc[idx])
    
    @classmethod
    def from_carray(cls, bcolz_array, labels, metadata, tfm_params=None, items=None, **kwargs):
        if items is None: items = range(len(bcolz_array))
        return cls(items, bcolz_array, labels, metadata, tfm_params, **kwargs)
    
    def analyze_pred(self, pred, thresh:float=0.5): return (pred[0].argmax(0), pred[1].argmax())
    
    def reconstruct(self, t:Tensor): return self._item_cls(t[0], t[1], self.classes[t[1]])
    
class ParallelFloatList(VolumeItemList):
    _item_cls = VolumeSegmentFloat
    def __init__(self, items, bcolz_array:bcolz.carray, labels, metadata:pd.DataFrame, tfm_params=None, **kwargs):
        super().__init__(items, bcolz_array, metadata, tfm_params, **kwargs)
        self.labels,self.loss_func = labels,ParallelLoss(regression=True)
    
    def new(self, items, **kwargs):
        return super(VolumeItemList, self).new(items, bcolz_array=self.bcolz_array, labels=self.labels,
                                               metadata=self.metadata, tfm_params=self.tfm_params, **kwargs)
    
    def get(self, i):
        idx = self.items[i]
        return self._item_cls(self.bcolz_array[idx], self.labels[idx], idx=idx, metadata=self.metadata.iloc[idx])
    
    @classmethod
    def from_carray(cls, bcolz_array, labels, metadata, tfm_params=None, items=None, **kwargs):
        if items is None: items = range(len(bcolz_array))
        return cls(items, bcolz_array, labels, metadata, tfm_params, **kwargs)
    
    def analyze_pred(self, pred, thresh:float=0.5): return (pred[0].argmax(0), pred[1])
    
    def reconstruct(self, t:Tensor): return self._item_cls(t[0], t[1])

class ParallelItemList(SegmentationItemList):
    @classmethod
    def from_paths(cls, path, path_masks, metadata_path, tfm_params=None):
        '''
        Use ex:
        data = (fastai_scans.ParallelItemList.from_paths(path/'train_scans128', path/'train_segmentation', path/'train_metadata.csv')
                                             .split_by_rand_pct(0.2, seed=7)
                                             .label_from_bcolz('mort')
                                             .transform(fastai_scans.get_transforms(), tfm_y=True)
                                             .databunch(bs=8)
                                             .normalize())
        '''
        return cls.from_carray(bcolz.open(path, mode='r'), bcolz.open(path_masks, mode='r'),
                               pd.read_csv(metadata_path), tfm_params)
    
    def label_from_bcolz(self, col_name, from_item_lists=False, **kwargs):
        labels = self.metadata[col_name].tolist()
        _label_cls = ParallelFloatList if isinstance(labels[0], (float, np.float32)) else ParallelLabelList
        y = _label_cls.from_carray(self.bcolz_array_masks, labels, self.metadata, self.tfm_params,
                                        items=self.items, path=self.path, **kwargs)
        res = self._label_list(x=self, y=y)
        return res
    
    def show_xys(self, xs, ys, channel=0, n_slices=4, imgsize=4, figsize=None,
                 x_cmap='magma', y_cmap='magma', slice_info=False, **kwargs):
        rows = len(xs)
        axs = seg_subplots(rows, n_slices, n_segs=2, imgsize=imgsize, figsize=figsize)
        for i, row_axs in enumerate(axs):
            xs[i].show(axes=row_axs[::2], label=None, cmap=x_cmap, channel=channel, **kwargs)
            ys[i].show(axes=row_axs[1::2], label=None, cmap=y_cmap, slice_info=slice_info, channel=channel, **kwargs)
        plt.tight_layout()
        
    def show_xyzs(self, xs, ys, zs, channel=0, n_slices=3, imgsize=4, figsize=None,
                  x_cmap='magma', y_cmap='magma', slice_info=False, **kwargs):
        rows = len(xs)
        axs = seg_subplots(rows, n_slices, n_segs=3, imgsize=imgsize, figsize=figsize)
        for i, row_axs in enumerate(axs):
            xs[i].show(axes=row_axs[::3], label=None, cmap=x_cmap, channel=channel, **kwargs)
            ys[i].show(axes=row_axs[1::3], label=None, cmap=y_cmap, slice_info=slice_info, channel=channel, **kwargs)
            zs[i].show(axes=row_axs[2::3], label=None, cmap=y_cmap, slice_info=slice_info, channel=channel, label_str='Pred', **kwargs)
        plt.tight_layout()
        
def seg_subplots(rows, cols, n_segs=2, imgsize=4, figsize=None, title=None, **kwargs):
    figsize = ifnone(figsize, (imgsize*cols, imgsize*rows*.6))
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    grid = fig.add_gridspec(rows, cols)
    axes = []
    for g in grid:
        inner_grid = g.subgridspec(1, n_segs, wspace=0.0, hspace=0.0)
        for ig in inner_grid:
            ax = plt.Subplot(fig, ig)
            fig.add_subplot(ax)
            axes.append(ax)
    if title is not None: fig.suptitle(title, **kwargs)
    return list(chunks(axes, cols*n_segs))

