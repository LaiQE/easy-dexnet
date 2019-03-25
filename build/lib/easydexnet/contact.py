import numpy as np
import logging


class Contact(object):
    """ 接触点类型，表示一个接触点
        主要用来计算接触点的摩擦锥
    """

    def __init__(self, point, normal, grasp_direction, moment_arm=None, config=None):
        """
        point: 接触点在物体坐标系的坐标点
        normal: 接触点所在面片的法线, 方向向外
        grasp_direction: 这个接触点的力作用方向, 方向向内
        moment_arm: 力臂,一个向量(接触点坐标 - 重心坐标)
        """
        self._point = point
        self._normal = normal / np.linalg.norm(normal)
        self._grasp_direction = grasp_direction / \
            np.linalg.norm(grasp_direction)
        self._moment_arm = moment_arm
        self._friction_cone = 0.5
        if config is not None:
            self._friction_cone = config['default_friction_coef']
        self._num_cone_faces = 8
        if config is not None:
            self._num_cone_faces = config['num_cone_faces']

    @property
    def point(self):
        return self._point

    @property
    def normal(self):
        return self._normal

    @property
    def grasp_direction(self):
        return self._grasp_direction

    def tangents(self, direction=None):
        """ 计算接触点的切线向量, 方向向量和切线向量在右手坐标系下定
            优化了原Dex-Net中的算法
        Parameters
        ----------
        direction : 3个元素的矩阵,用以计算与这个方向正交的平面

        Returns
        -------
        direction : 方向向量,如果未指定则为法向量的反方向
        t1 : 第一个切线向量
        t2 : 第二个切线向量
        """
        if direction is None:
            direction = -self._normal

        if np.dot(self._normal, direction) > 0:
            direction = -1 * direction

        x = np.array([1, 0, 0])
        # 计算x轴在切平面上的投影,作为第一个切向量
        v = x - direction*(x.dot(direction)/direction.dot(direction))
        w = np.cross(direction, v)

        v = v / np.linalg.norm(v)
        w = w / np.linalg.norm(w)

        return direction, v, w

    def friction_cone(self, num_cone_faces=None, friction_coef=None):
        """ 计算接触点处的摩擦锥.

        Parameters
        ----------
        num_cone_faces : int,摩擦锥近似的面数
        friction_coef : float,摩擦系数

        Returns
        -------
        success : bool,摩擦锥计算是否成功
        cone_support : 摩擦锥的边界向量
        normal : 向外的法向量
        """
        if num_cone_faces is None:
            num_cone_faces = self._num_cone_faces
        if friction_coef is None:
            friction_coef = self._friction_cone
        
        # 这里不能加, 每次计算的摩擦锥可能都不一样, 摩擦系数可能变
        # if self._friction_cone is not None and self._normal is not None:
        #     return True, self._friction_cone, self._normal

        # 获取切向量
        in_normal, t1, t2 = self.tangents()

        # 检查是否有相对滑动, 即切线方向的力始终大于摩擦力
        grasp_direction = self._grasp_direction
        normal_force_mag = np.dot(grasp_direction, in_normal)   # 法线方向分力
        tan_force_x = np.dot(grasp_direction, t1)               # t1方向分力
        tan_force_y = np.dot(grasp_direction, t2)               # t2方向分力
        tan_force_mag = np.sqrt(tan_force_x**2 + tan_force_y**2)  # 切平面上的分力
        friction_force_mag = friction_coef * normal_force_mag   # 最大静摩擦

        # 如果切面方向力大于最大静摩擦, 则生成失败
        if friction_force_mag < tan_force_mag:
            return False, None, self._normal

        # 计算摩擦锥
        force = in_normal
        cone_support = np.zeros((3, num_cone_faces))
        for j in range(num_cone_faces):
            tan_vec = t1 * np.cos(2 * np.pi * (float(j) / num_cone_faces)) + \
                t2 * np.sin(2 * np.pi * (float(j) / num_cone_faces))
            cone_support[:, j] = force + friction_coef * tan_vec

        # self._friction_cone = cone_support
        return True, cone_support, self._normal

    def torques(self, forces):
        """求出接触点上一组力矢量所能施加的力矩
        forces : 3xN 力矢量
        Returns: 3xN 一组力矩
        """
        num_forces = forces.shape[1]
        torques = np.zeros([3, num_forces])
        moment_arm = self._moment_arm
        for i in range(num_forces):
            torques[:, i] = np.cross(moment_arm, forces[:, i])
        return True, torques
    
    def normal_force_magnitude(self):
        """ 计算法线方向上的力的大小
        """
        normal_force_mag = 1.0
        if self._grasp_direction is not None and self._normal is not None:
            in_normal = -self._normal
            in_direction_norm = self._grasp_direction
            normal_force_mag = np.dot(in_direction_norm, in_normal)
        return max(normal_force_mag, 0.0)
