#!/usr/bin/env python
#-*- coding:utf-8 -*-
from .mesh import BaseMesh
from .stable_poses import StablePoses

class DexObject():
    def __init__(self, mesh, poses=None, grasps=None):
        self._mesh = mesh
        if not poses:
            self._poses = self.get_poses(self._mesh)
        if not grasps:
            pass
    
    @staticmethod
    def get_poses(mesh):
        if not isinstance(mesh, (BaseMesh)):
            raise TypeError('mesh must be the class BaseMesh')
        if mesh.is_watertight:
            _center_mass = mesh.center_mass
        else:
            _center_mass = mesh.centroid
        _raw_poses = mesh.convex_hull.compute_stable_poses(center_mass=_center_mass)
        return StablePoses.from_raw_poses(_raw_poses)
