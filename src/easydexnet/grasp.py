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
        self._config = config
        if width is None and config is not None:
            self._max_grasp_width = config['grispper']['max_width']
        self._min_grasp_width = min_width
        if min_width is None and config is not None:
            self._min_grasp_width = config['grispper']['min_width']
        self._alpha = 0.05
        if config is not None:
            self._alpha = config['grisp_distance_alpha']
        self._quality = None
    
    @property
    def quality(self):
        return self._quality
    
    @quality.setter
    def quality(self, q):
        self._quality = q

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
            return False, None

        is_c0, c0 = self._find_contact(mesh, point0, self._center)
        is_c1, c1 = self._find_contact(mesh, point1, self._center)
        if not (is_c0 and is_c1):
            logging.debug('close_fingers 接触点寻找失败'+str(is_c0)+str(is_c0))
            return False, None
        contacts = [c0, c1]

        return True, contacts

    def _check_approch(self, mesh, point, axis, dis):
        """ 检查路径上是否有碰撞
        mesh : 网格
        point : 检查点
        axis : 检查的方向,检查(point,point+axis*dis)
        dis : 检查的距离 """
        points, _ = mesh.intersect_line(point, point+axis*dis)
        if points.shape[0] > 0:
            return False
        return True

    def get_approch(self, poses):
        """ 计算在抓取路径, 即与z轴最近的方向 """
        poses_r = poses.matrix[:3, :3]
        # 稳定姿态的z轴在物体坐标下的表示
        poses_z = poses_r[2, :]
        axis = self._axis
        # 计算z轴在抓取轴为法线的平面上的投影作为抓取路径的反方向
        approach = poses_z - axis*(poses_z.dot(axis)/axis.dot(axis))
        approach_L = np.linalg.norm(approach)
        poses_z_L = np.linalg.norm(poses_z)
        angle = np.arccos(poses_z.dot(approach)/(approach_L*poses_z_L))
        angle = np.rad2deg(angle)
        approach = -approach / np.linalg.norm(approach)
        return approach, angle

    def check_approach(self, mesh, poses, config):
        """ 检查夹取路径上是否有碰撞 """
        # 获取抓取路径
        approch, _ = self.get_approch(poses)
        check_cfg = config['collision_checker']
        # 夹爪宽度方向
        width_axis = np.cross(approch, self._axis)
        w_axis = width_axis / np.linalg.norm(width_axis)
        check_num = max(check_cfg['checker_point_num'], 3)
        axis_offiset = check_cfg['axis_offiset']
        width_offset = check_cfg['width_offset']
        # 轴向补偿列表
        aixs_list = np.linspace(-axis_offiset/2, axis_offiset/2, check_num)
        # 宽度补偿列表
        width_list = np.linspace(-width_offset/2, width_offset/2, check_num)

        for p in self.endpoints:
            axis = self._axis
            # 最终检查列表
            check_list = [p + a*axis + w * w_axis
                          for a in aixs_list for w in width_list]
            for check_point in check_list:
                result = self._check_approch(mesh, check_point,
                                             -approch, check_cfg['test_dist'])
                if not result:
                    return False
        return True

    def apply_transform(self, matrix):
        center = np.r_[self.center, 1]
        center = matrix.dot(center)[:3]
        axis = matrix[:3, :3].dot(self._axis)
        return Grasp_2f(center, axis, config=self._config)
    
    @staticmethod
    def from_configuration(configuration, config):
        if not isinstance(configuration, np.ndarray) or (configuration.shape[0] != 9 and configuration.shape[0] != 10):
            raise ValueError('Configuration must be numpy ndarray of size 9 or 10')
        if configuration.shape[0] == 9:
            min_grasp_width = 0
        else:
            min_grasp_width = configuration[9]
        if np.abs(np.linalg.norm(configuration[3:6]) - 1.0) > 1e-5:
            raise ValueError('Illegal grasp axis. Must be norm one')
        center = configuration[0:3]
        axis = configuration[3:6]
        width = configuration[6]
        return Grasp_2f(center, axis, width, min_grasp_width, config)
    
    def to_configuration(self):
        configuration = np.zeros(10)
        configuration[0:3] = self._center
        configuration[3:6] = self._axis
        configuration[6] = self._max_grasp_width
        configuration[7] = 0
        configuration[8] = 0
        configuration[9] = self._min_grasp_width
        return configuration
