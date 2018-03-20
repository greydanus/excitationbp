# excitationbp: visualizing how deep networks make decisions
# Sam Greydanus. July 2017. MIT License.

from setuptools import setup, find_packages

setup(name='excitationbp',
      version='0.1',
      description='A minimal implementation of excitation backprop for PyTorch',
      author='Sam Greydanus',
      author_email=['greydanus.17@gmail.com'],
      license='MIT',
      packages=find_packages(),
      keywords=['deep-learning', 'excitation-backprop', 'visualization', 'interpretability'],
      zip_safe=False)