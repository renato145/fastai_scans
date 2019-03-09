from fastai.basics import *
import bcolz, nibabel as nib

def get_nidata(file): return np.rollaxis(nib.load(str(file)).get_data(), axis=2)

def get_brats_data(path, train_data=True):
    x_files = [path / f'{path.name}_{e}.nii.gz' for e in ['flair', 't1', 't1ce', 't2']]
    data_x = np.stack([get_nidata(f) for f in x_files], axis=0)
    data_y = None
    if train_data:
        y_file = path / f'{path.name}_seg.nii.gz'
        data_y = get_nidata(path / f'{path.name}_seg.nii.gz')
        data_y[data_y==4] = 3

    return data_x,data_y

class Processor():
    def func(self, volume, is_lbl=False): return volume
    def __call__(self, *item): return [vol if vol is None else self.func(vol,is_lbl) for vol,is_lbl in zip(item, [False,True]) if item]

class CropProcessor(Processor):
    def refresh(self, volume):
        tmp = volume.sum(0) if len(volume.shape) > 3 else volume
        x = np.where(tmp.sum((1,2))>0)[0]
        y = np.where(tmp.sum((0,2))>0)[0]
        z = np.where(tmp.sum((0,1))>0)[0]
        self.data = dict(x_min=x.min(), x_max=x.max(),
                         y_min=y.min(), y_max=y.max(),
                         z_min=z.min(), z_max=z.max())

    def func(self, volume, is_lbl=True):
        if not is_lbl: self.refresh(volume)
        d = self.data
        return volume[..., d['x_min']:d['x_max'], d['y_min']:d['y_max'], d['z_min']:d['z_max']]

class ResizeProcessor(Processor):
    def __init__(self, size):
        self.size = size

    def func(self, volume, is_lbl=True):
        is_int = issubclass(volume.dtype.type, np.integer)
        volume = tensor(volume[None]).float()
        if len(volume.shape) < 5: volume = volume[None]
        volume = F.interpolate(volume, size=self.size)[0]
        if volume.size(0) == 1: volume = volume[0]
        if is_lbl or is_int   : volume = volume.long()
        return to_np(volume)

def preprocess_brats(data_path, path, size, train_data=True, x_name='data', y_name='labels', csv_name='mapping.csv'):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    size = listify(size, 3)
    volumes = bcolz.zeros([0,4,*size], dtype=np.int64, chunklen=1, mode='w', rootdir=path / x_name)
    if train_data: labels  = bcolz.zeros([0,*size],   dtype=np.int64, chunklen=1, mode='w', rootdir=path / y_name)
    processors = [CropProcessor(), ResizeProcessor(size)]
    files = (data_path/'LGG').ls() + (data_path/'HGG').ls() if train_data else data_path.ls()

    (path / csv_name).open('w').write('modal,subject\n' if train_data else 'subject\n')
    with (path / csv_name).open('a') as f:
        for file in progress_bar(files):
            if not file.is_dir(): continue
            x,y = get_brats_data(file, train_data)
            for p in processors: x,y = p(x,y)
            volumes.append(x)
            if train_data: labels.append(y)
            f.write(f'{file.parent.name},{file.name}\n' if train_data else f'{file.name}\n')

    volumes.flush()
    if train_data: labels.flush()
