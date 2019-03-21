""" 渲染深度图的模块
    这里使用pyrender包进行渲染
    渲染时最重要的就是注意相机的位置和物体的位置
"""
import logging
import numpy as np
import scipy.stats as ss
import pyrender
from .camare import RandomCamera


class RenderScene(pyrender.Scene):
    """ 渲染器的场景, 从pyrender中继承
    """

    def add_obj(self, mesh, matrix=np.eye(4), offset=False):
        """ 添加一个物体到渲染的环境中
        mesh : BashMesh类型的对象
        matrix : (4,4)的变换矩阵
        offset : 补偿物体位置，在xOy平面上平移物体，使中心与原点对齐
        """
        matrix_ = matrix.copy()
        if offset:
            center = mesh.center_mass
            off = matrix_.dot(np.r_[center, 1.0])[:2]
            matrix_[:2, 3] = matrix_[:2, 3] - off
        tri = mesh.tri_mesh
        render_mesh = pyrender.Mesh.from_trimesh(tri)
        n = pyrender.Node(mesh=render_mesh, matrix=matrix_)
        self.add_node(n)
        return matrix_

    def add_camera(self, camrea):
        """ 向场景中添加一个相机
        camrea_matrix : 相机相对于世界坐标系的变换矩阵
        model : 相机模型参数数组
        """
        model = camrea.model
        yfov_ = model[0]
        aspectRatio_ = model[2]
        znear_ = model[1]
        camera_ = pyrender.PerspectiveCamera(
            yfov=yfov_, aspectRatio=aspectRatio_, znear=znear_)
        self.add(camera_, pose=camrea.pose)
        return camera_.get_projection_matrix()


class ImageRender(object):
    """ 保存了一副完整的深度图和创建深度图时环境的类 
    如果没有则渲染在一组给定的相机参数和物体位姿下的深度图
    """

    def __init__(self, mesh, pose, table, config, camera=None, data=None):
        if 'camera' in config.keys():
            config = config['camera']
        self._width = config['im_width']
        self._height = config['im_height']
        self._mesh = mesh
        self._pose = pose
        self._camera = camera
        self._table = table
        self._data = data
        self._camera_projection_matrix = None
        self._obj_matrix = None
        if camera is None:
            self._camera = RandomCamera(config)
        if data is None:
            self._data = self.render_image()

    @property
    def data(self):
        return self._data

    def render_image(self):
        scene = RenderScene()
        scene.add_obj(self._table)
        self._obj_matrix = scene.add_obj(self._mesh, self._pose.matrix, True)
        self._camera_projection_matrix = scene.add_camera(self._camera)
        r = pyrender.OffscreenRenderer(self._width, self._height)
        rgb, depth = r.render(scene)
        return [rgb, depth]

    def render_obj_point(self, point):
        """ 计算物体坐标系下的点在该图片下的特征向量
        [x, y, z, depth] """
        pose = self._camera.pose
        world_to_camera = np.linalg.inv(pose)
        point_in_world = self._obj_matrix.dot(np.r_[point, 1.0])
        point_in_camera = world_to_camera.dot(point_in_world)
        point_in_image = self._camera_projection_matrix.dot(
            point_in_camera)
        return point_in_image


class DepthRender(object):
    def __init__(self, dex_obj):
        pass

    @staticmethod
    def funcname(parameter_list):
        pass

    @staticmethod
    def render_stable(pose, dex_obj, config):
        """ 渲染一个稳定姿态下的所有深度图
        1. 设置物体和桌面的位置
        2. 设置相机的位置(随机生成多组相机位置)
        3. 生成深度图和所有夹爪在图像坐标下的位置
        4. 进行图像变换得到最终输出
        """
        pass
