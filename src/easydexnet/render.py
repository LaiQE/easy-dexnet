""" 渲染深度图的模块
    这里使用pyrender包进行渲染
    渲染时最重要的就是注意相机的位置和物体的位置
"""
import logging
import numpy as np
import scipy.stats as ss
import pyrender
import cv2
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
        self._viewport = self.get_viewport(self._size)

    @property
    def camera(self):
        return self._camera

    @property
    def data(self):
        return self._data

    @property
    def depth(self):
        return self._data[1]

    def get_viewport(self, size):
        """ 计算视口变换矩阵 """
        scale = np.array([[size[0]/2, 0, 0],
                          [0, size[1]/2, 0],
                          [0, 0, 1]])
        map_m = np.array([[1, 0, 1],
                          [0, -1, 1],
                          [0, 0, 1]])
        return scale.dot(map_m)

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


class DataGenerator(object):
    """ 数据生成器，从图像中生成最终的训练样本
    1. 平移夹爪中心到图像中心
    2. 旋转夹爪抓取轴与x轴对齐
    3. 裁剪到初步大小
    4. 缩放到最终大小
    注意: Dex-Net源码中这里夹爪宽度并没有与图像对齐,有待改进
    """

    def __init__(self, image, grasp_2d, config):
        self._image = image
        self._grasp = grasp_2d
        if "gqcnn" in config.keys():
            self._config = config["gqcnn"]

    @property
    def output(self):
        grasp = self._grasp
        crop_size = [self._config['crop_width'], self._config['crop_height']]
        out_size = [self._config['final_width'], self._config['final_height']]
        image = self.transform(self._image, grasp.center_float, grasp.angle)
        return self.crop_resize(image, crop_size, out_size)

    @staticmethod
    def transform(image, center, angle):
        """ 先把图片平移到给定点，再旋转给定角度
        注意:图片保存时维度0是行(即y轴)，维度1是列(即x轴)
        """
        angle_ = np.rad2deg(angle)
        image_size = np.array(image.shape[:2][::-1]).astype(np.int)
        translation = image_size / 2 - center
        trans_map = np.c_[np.eye(2), translation]
        rot_map = cv2.getRotationMatrix2D(
            tuple(image_size / 2), angle_, 1)
        trans_map_aff = np.r_[trans_map, [[0, 0, 1]]]
        rot_map_aff = np.r_[rot_map, [[0, 0, 1]]]
        full_map = rot_map_aff.dot(trans_map_aff)
        full_map = full_map[:2, :]
        im_data_tf = cv2.warpAffine(image, full_map, tuple(
            image_size), flags=cv2.INTER_NEAREST)
        return im_data_tf

    @staticmethod
    def crop_resize(image, crop_size, out_size, center=None):
        if center is None:
            center = (np.array(image.shape[:2][::-1]) - 1) / 2
        diag = np.array(crop_size) / 2
        start = center - diag
        end = center + diag
        image_crop = image[int(start[1]):int(
            end[1]), int(start[0]):int(end[0])].copy()
        image_out = cv2.resize(
            image_crop, (int(out_size[0]), int(out_size[0])))
        return image_out


class DepthRender(object):
    def __init__(self, dex_obj, table, saver, config):
        self._dex_obj = dex_obj
        self._config = config
        self._table = table
        self._saver = saver

    @property
    def dex_obj(self):
        return self._dex_obj

    def render(self):
        mesh = self._dex_obj.mesh
        table = self._table
        config = self._config
        saver = self._saver
        render_num = config['render']['images_per_stable_pose']
        # 对每个姿势迭代
        for pose in self._dex_obj.poses:
            vaild_grasps, collision = self.vaild_grasps(pose)
            for _ in range(render_num):
                render = ImageRender(mesh, pose, table, config)
                depth = render.depth
                # 对每个抓取
                for g, col in zip(vaild_grasps, collision):
                    quality = g.quality
                    if quality is None:
                        logging.error('抓取品质为None')
                    g_2d = render.render_grasp(g)
                    out = DataGenerator(depth, g_2d, config).output
                    saver.add(out, g_2d, quality, col)

    def vaild_grasps(self, pose):
        """ 获取该位姿下所有有效的夹爪，和是否碰撞
        """
        max_angle = self._config['render']['max_grasp_approch']
        mesh = self._dex_obj.mesh
        grasps = self._dex_obj.grasps
        vaild_g = []
        collision = []
        for g in grasps:
            if g.get_approch(pose)[1] < max_angle:
                vaild_g.append(g)
                collision.append(g.check_approach(mesh, pose, self._config))
        return vaild_g, collision
