import os
import io
import re
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


readme = read('README.md')

VERSION = '0.1.1'

requirements = [
    "torch==0.4.1",
    "torchvision==0.2.1",
    "tqdm>=4.19.5",
    "tensorboardX>=1.1",
    "ipython>=6.2.1",
    "opencv-python>=3.4.0.12",
    "Pillow>=5.0.0",
    "numpy>=1.14.0",
    "scikit-image>=0.13.1",
    "visdom>=0.1.8.5"
    "Cython==0.29.1",
    "pycocotools==2.0.0"
]

setup(
    # Metadata
    name='pysemseg',
    version=VERSION,
    author='Petko Nikolov',
    author_email='py.nikolov@gmail.com',
    url='https://github.com/petko-nikolov/pysemseg',
    description='Pytorch library for training Deep Learning models forSemantic Segmentation',
    long_description=readme,
    license='BSD',

    # Package info
    packages=find_packages(),

    zip_safe=True,
    install_requires=requirements,
)
