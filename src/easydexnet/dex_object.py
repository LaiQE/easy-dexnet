#!/usr/bin/env python
# -*- coding:utf-8 -*-
from .mesh import BaseMesh
from .stable_poses import StablePoses


class DexObject(object):
    """ 用于管理所有数据的类，其中包含了计算dex-net数据集中一个物体对象
        所需的所有数据，包括，网格数据、所有稳定姿态、所有候选抓取点
    """

    def __init__(self, mesh, poses=None, grasps=None):
        self._mesh = mesh
        if not poses:
            self._poses = self.get_poses(self._mesh)
        if not grasps:
            self._grasps = self.get_grasps(self._mesh)

    @staticmethod
    def get_poses(mesh):
        """ 从网格对象中获取所有稳定位姿 """
        if not isinstance(mesh, (BaseMesh)):
            raise TypeError('mesh must be the class BaseMesh')
        tri_mesh = mesh.tri_mesh
        if tri_mesh.is_watertight:
            _center_mass = tri_mesh.center_mass
        else:
            _center_mass = tri_mesh.centroid
        raw_poses = tri_mesh.convex_hull.compute_stable_poses(
            center_mass=_center_mass)
        return StablePoses.from_raw_poses(raw_poses)

    @staticmethod
    def get_grasps(mesh):
        """ 从网格对象中生成所有的候选抓取点 """
        # TODO 这是一个未完成的函数
        return mesh
