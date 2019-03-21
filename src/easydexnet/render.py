""" 渲染深度图的模块
    这里使用pyrender包进行渲染
    渲染时最重要的就是注意相机的位置和物体的位置
"""
import logging
import numpy as np
import scipy.stats as ss
import pyrender
from trimesh.visual.color import hex_to_rgba
from .camare import RandomCamera
from .grasp_2d import Grasp2D
from .colcor import cnames
from .vision import DexScene


class RenderScene(DexScene):
    """ 渲染器的场景, 从pyrender中继承
    """

    def add_obj(self, mesh, matrix=np.eye(4), offset=False, color=None):
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
        if color is not None:
            if isinstance(color, (str)):
                color = hex_to_rgba(cnames[color])
            color[-1] = 200
            tri.visual.face_colors = color
            tri.visual.vertex_colors = color
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

    def add_light(self, pose):
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        # light = pyrender.SpotLight(color=np.ones(
        #     3), intensity=3.0, innerConeAngle=np.pi/16.0, outerConeAngle=np.pi/6.0)
        self.add(light, pose=pose)


class ImageRender(object):
    """ 保存了一副完整的深度图和创建深度图时环境的类 
    如果没有则渲染在一组给定的相机参数和物体位姿下的深度图
    """

    def __init__(self, mesh, pose, table, config, camera=None, data=None):
        if 'camera' in config.keys():
            config = config['camera']
        self._size = np.array(
            [config['im_width'], config['im_height']], np.int)
        self._obj_color = config['obj_color']
        self._table_color = config['table_color']
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
        # 视口变换矩阵
        self._viewport = np.array([[self._size[0]/2,0,self._size[0]/2],
                                   [0,-self._size[1]/2,self._size[1]/2],
                                   [0,0,1]]) 

    @property
    def data(self):
        return self._data

    @property
    def depth(self):
        return self._data[1]

    def render_image(self):
        scene = RenderScene()
        scene.add_obj(self._table, color=self._table_color)
        self._obj_matrix = scene.add_obj(
            self._mesh, self._pose.matrix, True, color=self._obj_color)
        self._camera_projection_matrix = scene.add_camera(self._camera)
        scene.add_light(self._camera.pose)
        self.scene = scene
        r = pyrender.OffscreenRenderer(self._size[0], self._size[1])
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

    def render_grasp(self, grasp_3d):
        """ openGl在进行渲染时的过程参考下面的博客
            https://blog.csdn.net/linuxheik/article/details/81747087
            https://blog.csdn.net/wangdingqiaoit/article/details/51589825
            这里再上面得到的NDC坐标之后还要进行视口变换
            由于最终的图像是Y轴朝下的，而这里的NDC坐标是Y轴朝上
            为了保证最终的图像不变，这里的y坐标全部要取反
        """
        p0_3d, p1_3d = grasp_3d.endpoints
        p0_2d = self.render_obj_point(p0_3d)
        p1_2d = self.render_obj_point(p1_3d)
        p0_2d = p0_2d / p0_2d[-1]
        p1_2d = p1_2d / p1_2d[-1]
        p0_in_image = self._viewport.dot(np.r_[p0_2d[:2], 1.])[:2]
        p1_in_image = self._viewport.dot(np.r_[p1_2d[:2], 1.])[:2]
        center_3d = grasp_3d.center
        center_2d = self.render_obj_point(center_3d)
        v = np.r_[p0_in_image, p1_in_image, center_2d[-1]]
        return Grasp2D.from_feature_vec(v)


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
