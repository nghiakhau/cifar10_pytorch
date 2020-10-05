from setuptools import setup, find_packages

with open('README.md', encoding='utf8') as f:
    readme = f.read()

setup(
    name='cifar10',
    version='0.1.0',
    description='Train CIFAR 10 dataset with PyTorch',
    packages=find_packages()
)