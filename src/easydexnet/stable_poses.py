#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np


class StablePoses(object):
    def __init__(self, matrix, probability):
        self._matrix = matrix
        self._probability = probability

    @staticmethod
    def from_raw_poses(raw_poses):
        poses = []
        for matrix, probability in zip(*raw_poses):
            poses.append(StablePoses(matrix, probability))
        return poses
