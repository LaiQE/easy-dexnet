#!/usr/bin/env python
# -*- coding:utf-8 -*-
import logging
import os.path
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
        for q, g in zip(self._qualitis, self._grasps):
            g.quality = q

    @property
    def name(self):
        return self._name

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
        if not name:
            name = os.path.splitext(os.path.basename(file_path))[0]
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
        return DexObject(mesh, config, poses, grasps, metrics, name)

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
            # print(m)
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
            if r.shape[0] == 3:
                v = r.dot(vertices.T)
                min_z = np.min(v[2, :])
                matrix = np.eye(4)
                matrix[:3, :3] = r
                matrix[2, 3] = -min_z
            else:
                matrix = r
            poses.append(StablePoses(matrix, p))
        return poses

    def to_hdf5_group(self, obj_group, config):
        if self._name in list(obj_group.keys()):
            del obj_group[self._name]
        group = obj_group.require_group(self._name)
        self.grasps_to_hdf5(group, config)
        self.mesh_to_hdf5(group)
        self.poses_to_hdf5(group, config)

    def grasps_to_hdf5(self, obj_group, config):
        group_name = config['hdf5_config']['grasps_group']
        metrics_name = config['hdf5_config']['metrics_name']
        group = obj_group.require_group(group_name)
        for i in range(len(self._grasps)):
            configuration = self._grasps[i].to_configuration()
            metrics = self._qualitis[i]
            grasp_name = 'grasp_%d' % (i)
            grasp_group = group.require_group(grasp_name)
            grasp_group.attrs['configuration'] = configuration
            metrics_group = grasp_group.require_group('metrics')
            metrics_group.attrs[metrics_name] = metrics

    def mesh_to_hdf5(self, obj_group):
        triangles = self._mesh.tri_mesh.faces
        vertices = self._mesh.tri_mesh.vertices
        group = obj_group.require_group('mesh')
        group.create_dataset('triangles', data=triangles)
        group.create_dataset('vertices', data=vertices)

    def poses_to_hdf5(self, obj_group, config):
        group_name = config['hdf5_config']['stable_poses_group']
        group = obj_group.require_group(group_name)
        for i, pose in enumerate(self._poses):
            pose_name = 'pose_%d' % (i)
            pose_group = group.require_group(pose_name)
            pose_group.attrs['r'] = pose.matrix
            pose_group.attrs['p'] = pose.probability
