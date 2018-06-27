#nsml: pytorch/pytorch
from distutils.core import setup
setup(
    name='nsml SRMD',
    version='1.0',
    description='ns-ml',
    install_requires=[
        'torch==0.4.0',
        'visdom',
        'pillow',
        'h5py' 
    ]
)
