#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np


class StablePoses(object):
    def __init__(self, matrix, probability):
        self._matrix = matrix
        self._probability = probability
    
    @property
    def matrix(self):
        return self._matrix
    
    @property
    def probability(self):
        return self._probability
        
    @staticmethod
    def from_raw_poses(matrixs, probabilitys):
        poses = []
        for matrix, probability in zip(matrixs, probabilitys):
            poses.append(StablePoses(matrix, probability))
        return poses
