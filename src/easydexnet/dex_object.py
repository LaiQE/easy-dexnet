#!/usr/bin/env python
# -*- coding:utf-8 -*-
import logging
import numpy as np
from .mesh import BaseMesh
from .stable_poses import StablePoses
from .grasp_sampler import GraspSampler_2f
from .quality import grasp_quality
from .grasp import Grasp_2f


class DexObject(object):
    """ 用于管理所有数据的类，其中包含了计算dex-net数据集中一个物体对象
        所需的所有数据，包括，网格数据、所有稳定姿态、所有候选抓取点
    """

    def __init__(self, mesh, config, poses=None, grasps=None, qualitis=None, name=None):
        self._mesh = mesh
        self._config = config
        self._poses = poses
        self._grasps = grasps
        self._qualitis = qualitis
        self._name = name
        if not poses:
            self._poses = self.get_poses(
                self._mesh, config['stable_pose_min_p'])
        if not grasps:
            self._grasps = self.get_grasps(self._mesh, self._config)
        if not qualitis:
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
    def from_file(file_path, config, name=None):
        mesh = BaseMesh.from_file(file_path, name)
        return DexObject(mesh, config, name=name)

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

    @staticmethod
    def from_hdf5_group(obj_group, config, name=None):
        grasps, metrics = DexObject.grasps_from_hdf5(obj_group, config)
        mesh = DexObject.mesh_from_hdf5(obj_group, name)
        poses = DexObject.poses_from_hdf5(obj_group, config)
        return DexObject(mesh, config, poses, grasps, metrics)

    @staticmethod
    def grasps_from_hdf5(obj_group, config):
        group = obj_group[config['hdf5_config']['grasps_group']]
        metrics_name = config['hdf5_config']['metrics_name']
        grasps = []
        metrics = []
        for grasp_name, grasp_group in group.items():
            configuration = grasp_group.attrs['configuration']
            g = Grasp_2f.from_configuration(configuration, config)
            grasps.append(g)
            m = grasp_group['metrics'].attrs[metrics_name]
            metrics.append(m)
        return grasps, metrics

    @staticmethod
    def mesh_from_hdf5(obj_group, name):
        triangles = np.array(obj_group['mesh/triangles'])
        vertices = np.array(obj_group['mesh/vertices'])
        mesh = BaseMesh.from_data(vertices, triangles, name=name)
        return mesh

    @staticmethod
    def poses_from_hdf5(obj_group, config):
        group = obj_group[config['hdf5_config']['stable_poses_group']]
        vertices = np.array(obj_group['mesh/vertices'])
        poses = []
        for pose_name, pose in group.items():
            p = pose.attrs['p']
            r = pose.attrs['r']
            # x0 = pose.attrs['x0']
            v = r.dot(vertices.T)
            min_z = np.min(v[2,:])
            # 这里的x0什么作用还要试一下
            matrix = np.eye(4)
            matrix[:3, :3] = r
            matrix[2, 3] = -min_z
            poses.append(StablePoses(matrix, p))
        return poses
    
    def to_hdf5_group(self, parameter_list):
        pass
