#!/usr/bin/env python
# -*- coding:utf-8 -*-
import logging
from .mesh import BaseMesh
from .stable_poses import StablePoses
from .grasp_sampler import GraspSampler_2f
from .quality import grasp_quality


class DexObject(object):
    """ 用于管理所有数据的类，其中包含了计算dex-net数据集中一个物体对象
        所需的所有数据，包括，网格数据、所有稳定姿态、所有候选抓取点
    """

    def __init__(self, mesh, config, poses=None, grasps=None):
        self._mesh = mesh
        self._config = config
        if not poses:
            self._poses = self.get_poses(
                self._mesh, config['stable_pose_min_p'])
        if not grasps:
            self._grasps = self.get_grasps(self._mesh, self._config)
        self._qualitis, self._grasps = self.get_quality(
            self._grasps, self._mesh, self._config)

    @property
    def mesh(self):
        return self._mesh

    @property
    def poses(self):
        return self._poses

    @property
    def grasps(self):
        return self._grasps

    @property
    def qualitis(self):
        return self._qualitis

    @staticmethod
    def from_trimesh(tri_mesh, config, name=None):
        mesh = BaseMesh(tri_mesh, name)
        return DexObject(mesh, config)

    @staticmethod
    def get_poses(mesh, threshold):
        """ 从网格对象中获取所有稳定位姿 """
        if not isinstance(mesh, (BaseMesh)):
            raise TypeError('mesh must be the class BaseMesh')
        tri_mesh = mesh.tri_mesh
        _center_mass = mesh.center_mass
        matrixs, probabilitys = tri_mesh.convex_hull.compute_stable_poses(
            center_mass=_center_mass, threshold=threshold)
        return StablePoses.from_raw_poses(matrixs, probabilitys)

    @staticmethod
    def get_grasps(mesh, config):
        if not isinstance(mesh, (BaseMesh)):
            raise TypeError('mesh must be the class BaseMesh')
        sampler = GraspSampler_2f(config=config)
        num_sample = config['num_sample']
        grasps = sampler.generate_grasps(mesh, num_sample)
        return grasps

    @staticmethod
    def get_quality(grasps, mesh, config):
        metrics = config['metrics']
        valid_grasps = []
        qualitis = []
        for grasp in grasps:
            try:
                quality = grasp_quality(grasp, mesh, metrics)
            except Exception as e:
                logging.warning('抓取品质计算无效')
                print('抓取品质计算无效', e)
            else:
                qualitis.append(quality)
                valid_grasps.append(grasp)
        return qualitis, valid_grasps
