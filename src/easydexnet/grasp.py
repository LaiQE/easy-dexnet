import logging
import numpy as np
from .contact import Contact


class Grasp_2f(object):
    """ 点接触二指夹爪抓取点模型类
    """

    def __init__(self, center, axis, width, angle=0, jaw_width=0, min_width=0):
        """
        center: 夹爪的中心点
        axis: 夹爪的二指连线方向向量
        width: 最大张开距离
        angle: 夹爪的进入路径方向
        jaw_width: 夹爪的横向宽度
        min_width: 夹爪闭合时的宽度
        """
        self._center = center
        self._axis = axis
        self._max_grasp_width = width
        self._min_grasp_width = min_width
        self._jaw_width = jaw_width
        self._approach_angle = angle

    @property
    def center(self):
        return self._center

    @property
    def axis(self):
        return self._axis

    @property
    def endpoints(self):
        """ 返回夹爪的两个端点 """
        point0 = self._center - (self._max_grasp_width / 2.0) * self._axis
        point1 = self._center + (self._max_grasp_width / 2.0) * self._axis
        return point0, point1

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

    def close_fingers(self, mesh, check_approach=True, approach_dist=0.2):
        # TODO 检查在接近路径上的碰撞点,暂时先不写
        if check_approach:
            pass
        
        point0, point1 = self.endpoints
        # 先判断中间的交点是否为偶数并且是否有足够的点,奇数交点个数则表示出错
        points, _ = mesh.intersect_line(point0, point1)
        if ((points.shape[0]%2) != 0) or points.shape[0] < 2:
            return False, None, None
            
        is_c0, c0 = self._find_contact(mesh, point0, self._center)
        is_c1, c1 = self._find_contact(mesh, point1, self._center)
        if not (is_c0 and is_c1):
            return False, None, None

        return True, c0, c1

    @staticmethod
    def grasp_from_one_contact():
        pass
