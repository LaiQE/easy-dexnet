import logging
import numpy as np
from .contact import Contact


class Grasp_2f(object):
    """ 点接触二指夹爪抓取点模型类
    """

    def __init__(self, center, axis, width=None, min_width=None, config=None):
        """
        center: 夹爪的中心点
        axis: 夹爪的二指连线方向向量
        width: 最大张开距离
        min_width: 夹爪闭合时的宽度
        """
        self._center = center
        self._axis = axis / np.linalg.norm(axis)
        self._max_grasp_width = width
        if width is None and config is not None:
            self._max_grasp_width = config['grispper']['max_width']
        self._min_grasp_width = min_width
        if width is None and config is not None:
            self._min_grasp_width = config['grispper']['min_width']
        self._alpha = 0.05
        if config is not None:
            self._alpha = config['grisp_distance_alpha']

    @property
    def center(self):
        return self._center

    @property
    def axis(self):
        return self._axis

    @property
    def endpoints(self):
        """ 返回夹爪的两个端点 """
        half_axis = (self._max_grasp_width / 2.0) * self._axis
        point0 = self._center - half_axis
        point1 = self._center + half_axis
        return point0, point1
    
    @property
    def width(self):
        return self._max_grasp_width

    @staticmethod
    def distance(g1, g2, alpha=0.05):
        """ 计算两个夹爪对象之间的距离
        """
        center_dist = np.linalg.norm(g1.center - g2.center)
        axis_dist = (2.0 / np.pi) * np.arccos(np.abs(g1.axis.dot(g2.axis)))
        return center_dist + alpha * axis_dist

    def _find_contact(self, mesh, out_point, center_point):
        """ 找到一个接触点, 从外部点到中心点的第一个接触点
            mesh: 一个BaseMesh对象
            out_point: 夹爪的一个端点
            center_point: 夹爪的中心点
        """
        points, cell_ids = mesh.intersect_line(out_point, center_point)
        direction = center_point - out_point

        if points.shape[0] < 1:
            return False, None

        # 求取法向量, 这里的法向量要保证朝向外侧
        normal = mesh.tri_mesh.face_normals[cell_ids[0]]
        if np.dot(direction, normal) > 0:
            normal = -normal
        # 求取接触点的力臂,由质心指向接触点的向量
        moment_arm = points[0] - mesh.center_mass
        c = Contact(points[0], normal, direction, moment_arm)
        return True, c

    def close_fingers(self, mesh, check_approach=True):
        """ 闭合夹爪,返回接触点
            mesh: 一个BaseMesh对象
            check_approach: 是否进行碰撞检测
        """
        # TODO 检查在接近路径上的碰撞点,暂时先不写
        if check_approach:
            pass

        point0, point1 = self.endpoints
        # 先判断中间的交点是否为偶数并且是否有足够的点,奇数交点个数则表示出错
        points, _ = mesh.intersect_line(point0, point1)
        if ((points.shape[0] % 2) != 0) or points.shape[0] < 2:
            logging.debug('close_fingers 交点生成出错')
            return False, None, None

        is_c0, c0 = self._find_contact(mesh, point0, self._center)
        is_c1, c1 = self._find_contact(mesh, point1, self._center)
        if not (is_c0 and is_c1):
            logging.debug('close_fingers 接触点寻找失败'+str(is_c0)+str(is_c0))
            return False, None, None

        return True, (c0, c1)

    @staticmethod
    def grasp_from_one_contact():
        pass
