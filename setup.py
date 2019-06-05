# -*- coding: utf-8 -*-
"""
Setup of easy-dexnet python codebase
Author: Lai QE
"""
from setuptools import setup,find_packages

requirements = [
      'trimesh', 
      'pyrender',
      'opencv-python' 
]

setup(name='easy-dexnet',                       
      version='0.1.0',                         
      description='easy-dexnet project',
      author='LaiQE',
      author_email='laiqe@qq.com',
      package_dir = {'': 'src'},                
      packages=find_packages('src'),            
      install_requires=requirements,            
      # test_suite='test'
     )

