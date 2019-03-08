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
        point0 = self._center - (self._max_grasp_width / 2.0) * self._axis
        point1 = self._center + (self._max_grasp_width / 2.0) * self._axis
        return point0, point1

    def _find_contacts(self, mesh, point0, point1):
        points, cell_ids = mesh.intersect_line(point0, point1)
        if (len(points) % 2) > 0:
            return False, None, None
        # 指向point1的向量
        direction = point1 - point0

        # TODO 这里无法确保法线的方向为向外
        c0_normal = mesh.tri_mesh.face_normals[cell_ids[0]]
        c0_moment_arm = points[0] - mesh.center_mass
        c0 = Contact(points[0], c0_normal, direction, c0_moment_arm)

        c1_normal = mesh.tri_mesh.face_normals[cell_ids[-1]]
        c1_moment_arm = points[-1] - mesh.center_mass
        c1 = Contact(points[-1], c1_normal, -direction, c1_moment_arm)
        return True, c0, c1

    def close_fingers(self, mesh, check_approach=True, approach_dist=0.2):
        # TODO 检查在接近路径上的碰撞点,暂时先不写
        if check_approach:
            pass

        return self._find_contacts(mesh, *self.endpoints)

    @staticmethod
    def grasp_from_one_contact():
        pass
