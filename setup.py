# setup.py
from setuptools import setup, find_packages

setup(
    name='tkan',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'scipy'
    ]
)