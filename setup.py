from setuptools import setup, find_packages

setup(name='fastai_scans',
      version = '0.0.1',
      description = "Library for deep learning on 3d images using fastai and Pytorch",
      url = "https://github.com/renato145/fastai_scans",
      author = "Renato Hermoza Aragonés",
      packages = find_packages(),
      install_requires = ["fastai", "bcolz", "nibabel", "h5py", "ipyvolume"],
      )
