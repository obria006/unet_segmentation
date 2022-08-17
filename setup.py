from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Unet for tissue segmentation to be used with Autoinjector 2.0.',
    author="Jacob O'Brien",
    license='',
    install_requires=[
          'matplotlib',
          'numpy',
          'scipy',
          "tifffile",
          "numba",
          "tqdm",
          "jupyter",
          "pandas",
          "seaborn",
          "scikit-image",
          "colorspacious",
          "pytest",
          "imagecodecs",
          "torch",
          "torchvision",
          "czifile",
          "opencv-python"
        ]
)
