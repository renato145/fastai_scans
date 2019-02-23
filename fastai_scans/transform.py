from fastai.basics import *
from fastai.vision.image import _get_default_args

__all__ = ['get_transforms']

class Transform():
    order=0
    def __init__(self, func, order=None):
        if order is not None: self.order=order
        self.func = func
        self.func.__name__ = func.__name__[1:]
        self.def_args = _get_default_args(func)
        self.params = copy(func.__annotations__)
        
    def __call__(self, *args, is_random=True, p=1., **kwargs):
        if args: return self.calc(*args, **kwargs)
        else: return RandTransform(self, kwargs=kwargs, is_random=is_random, p=p)
        
    def calc(self, x, *args, **kwargs): return self.func(x, *args, **kwargs)
        
@dataclass
class RandTransform():
    "Wrap `Transform` to add randomized execution."
    tfm:Transform
    kwargs:dict
    p:float=1.0
    resolved:dict = field(default_factory=dict)
    do_run:bool = True
    is_random:bool = True
    def resolve(self, tfm_params=None):
        "Bind any random variables in the transform."
        if not self.is_random:
            self.resolved = {**self.tfm.def_args, **self.kwargs}
            return

        self.resolved = {}
        # solve patches tfm
        if tfm_params is not None:
            tfm_params = self.tfm.resolve(tfm_params)
            for k,v in tfm_params.items(): self.resolved[k] = v

        # for each param passed to tfm...
        for k,v in self.kwargs.items():
            # ...if it's annotated, call that fn...
            if k in self.tfm.params:
                rand_func = self.tfm.params[k]
                self.resolved[k] = rand_func(*listify(v))
            # ...otherwise use the value directly
            else: self.resolved[k] = v
        # use defaults for any args not filled in yet
        for k,v in self.tfm.def_args.items():
            if k not in self.resolved: self.resolved[k]=v

    @property
    def order(self): return self.tfm.order

    def __call__(self, x, *args, **kwargs):
        "Randomly execute our tfm on `x`."
        return self.tfm(x, *args, **{**self.resolved, **kwargs}) if self.do_run else x

class TfmCrop(Transform):
    order=0
class TfmZoom(Transform):
    order=1
class TfmPatch(Transform):
    order=-1
    def __init__(self, func, resolve, order=None):
        '''
        func must be a tfm function.
        resolve is a function with 'tfm_params' input and returns the arguments that
        will be used for the 'func' tfm.
        '''
        super().__init__(func, order)
        self.resolve = resolve

def get_transforms(do_crop=False, max_zoom=1.1, random_crop=True, validation_random_crop=False, xtra_tfms=None,
                   validation_xtra_tfms=None):
    res = []
    val_res = []
    if do_crop:
        res.append(rand_crop() if random_crop else crop())
        val_res.append(rand_crop() if validation_random_crop else crop())
    if max_zoom > 1: res.append(rand_zoom(scale=(1., max_zoom)))

    res += listify(xtra_tfms)
    val_res += listify(validation_xtra_tfms)
    return (res, val_res)

def _crop(x, size, x_pct:uniform=0.5, y_pct:uniform=0.5, z_pct:uniform=0.5):
    if size is None: return x
    size = listify(size,3)
    vol_size = x.shape[-3:]
    if vol_size == torch.Size(size): return x
    xs,ys,zs = [min(sz,vsz) for sz,vsz in zip(size,vol_size)]
    xx = int((x.size(-3)-xs+1) * x_pct)
    yy = int((x.size(-2)-ys+1) * y_pct)
    zz = int((x.size(-1)-zs+1) * z_pct)
    x = x[..., xx:xx+xs, yy:yy+ys, zz:zz+zs]
    return x.contiguous()

def _zoom(x, scale:uniform=1.0, x_pct:uniform=0.5, y_pct:uniform=0.5, z_pct:uniform=0.5, **kwargs):
    size = x.shape
    is_int = x.dtype == torch.int64
    x = F.interpolate(x[None].float(), scale_factor=scale)[0]
    return _crop(x.long() if is_int else x, size=size[-3:], x_pct=x_pct, y_pct=y_pct, z_pct=z_pct)

def _sample_patch(tfm_params):
    args = tfm_params[uniform_int(0,len(tfm_params)-1)]
    return dict(zip(['x_pct', 'y_pct', 'z_pct'], args))

crop = TfmCrop(_crop)
zoom = TfmZoom(_zoom)
sample_patch = TfmPatch(_crop, _sample_patch)

rand_pos = {'x_pct':(0,1), 'y_pct':(0,1), 'z_pct':(0,1)}

def rand_crop(*args, **kwargs):
    "Randomized version of `crop`."
    return crop(*args, **rand_pos, **kwargs)

def rand_zoom(*args, **kwargs):
    "Randomized version of `zoom`."
    return zoom(*args, **rand_pos, **kwargs)
