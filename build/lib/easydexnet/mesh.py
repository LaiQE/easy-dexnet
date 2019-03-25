#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import trimesh
import numpy as np
from tvtk.api import tvtk


class BaseMesh(object):
    ''' 基础的网格文件类，用以保存原始数据
    '''

    def __init__(self, trimesh_obj, name=None):
        '''
        trimesh_obj: 一个trimesh对象
        name: 文件的名字
        '''
        # 一个trimesh对象,用来保存网格数据
        self._trimesh_obj = self._process_mesh(trimesh_obj)
        # 一个tvtk中的obb对象,用来计算相交线
        self._obb_tree = self._generate_obbtree(self._trimesh_obj)
        self._name = name
        if self._trimesh_obj.is_watertight:
            self._center_mass = self._trimesh_obj.center_mass
        else:
            self._center_mass = self._trimesh_obj.centroid

    @property
    def tri_mesh(self):
        return self._trimesh_obj

    @property
    def center_mass(self):
        return self._center_mass

    @property
    def name(self):
        return self._name

    def intersect_line(self, lineP0, lineP1):
        ''' 计算与线段相交的交点，这里调用了tvtk的方法
        lineP0: 线段的第一个点，长度3的数组
        lineP1: 线段的第二个点
        return
        points: 所有的交点坐标
        cell_ids: 每个交点所属的面片ID '''
        intersectPoints = tvtk.to_vtk(tvtk.Points())
        intersectCells = tvtk.to_vtk(tvtk.IdList())
        self._obb_tree.intersect_with_line(
            lineP0, lineP1, intersectPoints, intersectCells)
        intersectPoints = tvtk.to_tvtk(intersectPoints)
        intersectCells = tvtk.to_tvtk(intersectCells)
        points = intersectPoints.to_array()
        cell_ids = [intersectCells.get_id(
            i) for i in range(intersectCells.number_of_ids)]
        return points, cell_ids

    def _generate_obbtree(self, trimesh_obj):
        ''' 用来生成一个可用的obbtree对象，加速后面相交的判断 '''
        poly = tvtk.PolyData(points=trimesh_obj.vertices,
                             polys=trimesh_obj.faces)
        tree = tvtk.OBBTree(data_set=poly, tolerance=1.e-8)
        tree.build_locator()
        return tree

    def _process_mesh(self, trimesh_obj):
        ''' 用来预处理mesh数据,这里只是简单的调用了trimesh的预处理程序 '''
        # TODO 可以补上对物体水密性的处理
        if not trimesh_obj._validate:
            trimesh_obj._validate = True
            trimesh_obj.process()
        return trimesh_obj

    def bounding_box(self):
        max_coords = np.max(self._trimesh_obj.vertices, axis=0)
        min_coords = np.min(self._trimesh_obj.vertices, axis=0)
        return min_coords, max_coords

    def apply_transform(self, matrix):
        tri = self._trimesh_obj.copy()
        tri = tri.apply_transform(matrix)
        return BaseMesh(tri, self._name)

    @staticmethod
    def from_file(file_path, name=None):
        if name is None:
            name = os.path.splitext(os.path.basename(file_path))[0]
        tri_mesh = trimesh.load_mesh(file_path, validate=True)
        return BaseMesh(tri_mesh, name)

    @staticmethod
    def from_data(vertices, triangles, normals=None, name=None):
        trimesh_obj = trimesh.Trimesh(vertices=vertices,
                                      faces=triangles,
                                      face_normals=normals,
                                      validate=True)
        return BaseMesh(trimesh_obj, name)
