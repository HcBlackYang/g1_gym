from setuptools import find_packages
from distutils.core import setup

setup(
    name='g1_gym',
    version='1.0.0',
    author='Blake Yang',
    license="BSD-3-Clause",
    packages=find_packages(),
    description='Reinforcement Learning environments for G1 Robot',
    install_requires=[
        'numpy',
        'torch',
        'tensorboard',
        'matplotlib',
        'pyyaml',
        'tqdm',
        'gym',
        'termcolor',
        'scipy'
    ],
    python_requires='>=3.8',
)