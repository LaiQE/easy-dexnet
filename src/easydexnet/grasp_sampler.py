import random
import logging
import numpy as np
from .grasp import Grasp_2f
from .contact import Contact
from .quality import force_closure_2f


class GraspSampler_2f(object):
    """ 二指夹爪抓取点采样器，采用对映采样器
    1. 随机均匀采样物体表面的点
    2. 在该点的摩擦锥内随机采样一个抓取方向
    3. 沿着这个方向采样另一个抓取点
    4. 检查抓取点
    """
    def __init__(self, width=None, min_contact_dist=None, config=None):
        """
        width : 夹爪的最大宽度
        min_contact_dist : 接触点之间容忍的最小距离
        friction_coef : 摩擦系数
        dist_thresh : 夹爪之间容忍的最小间距
        """
        self._max_grasp_width = width
        if width is None and config is not None:
            self._max_grasp_width = config['grispper']['max_width']
        self._min_contact_dist = min_contact_dist
        if min_contact_dist is None and config is not None:
            self._min_contact_dist = config['min_contact_dist']
        self._friction_coef = 0.5
        if config is not None:
            self._friction_coef = config['default_friction_coef']
        self._grasp_dist_thresh = 0.0005
        if config is not None:
            self._grasp_dist_thresh = config['grasp_dist_thresh']
        if config is not None:
            self._config = config

    def _sample_vector(self, contact, num_samples=1):
        """ 在给定点接触点的摩擦锥内采样随机的方向向量
        采样的方向随机向内或向外,这里朝向无所谓
        contract : 一个接触点
        friction_coef : 摩擦系数
        num_samples : 采样个数
        return : 方向向量列表
       """
        friction_coef = self._friction_coef
        n, tx, ty = contact.tangents()
        v_samples = []
        # TODO 这里的循环可以改成纯numpy的形式
        for _ in range(num_samples):
            theta = 2 * np.pi * np.random.rand()
            r = friction_coef * np.random.rand()
            v = n + r * np.cos(theta) * tx + r * np.sin(theta) * ty
            v = -v / np.linalg.norm(v)
            v_samples.append(v)
        return v_samples

    def _find_grasp(self, mesh, point, vector, test_dist=0.5):
        """ 利用给定的点和方向，找到另一个接触点,并生成抓取模型
        mesh : 待抓取的物体网格
        point : 已知的一个接触点
        vector : 给定的抓取方向
        test_dist : 用于测试的距离, 物体最大距离的一半
        ---
        is_grasp : 生成抓取模型是否成功
        grasp : 生成的抓取模型 
        """
        given_point = point
        given_point_index = -1
        # 先在大范围上计算出所有的由给定点和方向指定的直线
        p0 = given_point - vector*test_dist
        p1 = given_point + vector*test_dist
        points, _ = mesh.intersect_line(p0, p1)

        # 如果交点个数为奇数则出错
        points_len = points.shape[0]
        if points_len % 2 != 0 or points_len == 0:
            logging.debug('_find_grasp 交点数检查出错，物体:%s ' % (mesh.name))
            return False, None

        # 找到当前点的位置, 由于计算误差用一个极小数测试是否一致
        for i, p in enumerate(points):
            if np.linalg.norm(p - given_point) < 1.e-6:
                given_point_index = i
                break
        # 如果给定点没有在计算出的交点内，则出错
        if given_point_index == -1:
            logging.debug('_find_grasp 给定点未在物体表面，物体:%s ' % (mesh.name))
            return False, None
        # 生成候选的配对点
        if given_point_index % 2 == 0:
            candidate_point_index = list(
                range(given_point_index+1, points_len, 2))
            left_point_index = given_point_index - 1
            right_dir = 1
        else:
            candidate_point_index = list(range(given_point_index-1, -1, -2))
            left_point_index = given_point_index + 1
            right_dir = -1
        # 最后找到抓取列表
        grasps = []
        # 对于每一个候选配对点检查距离是否小于最小距离或是否会被下一个点挡住
        # TODO 这里可以写的更清晰一点
        for inedx in candidate_point_index:
            p = points[inedx]
            # 判断两个点之间的距离是否大于要求的最小距离
            distance = np.linalg.norm(given_point - p)
            distance_is = distance > self._min_contact_dist and distance < self._max_grasp_width
            center = (given_point + p) / 2
            # 判断夹爪的右边是否不会碰到
            right_point_index = inedx + right_dir
            if right_point_index >= points_len or right_point_index < 0:
                right_is = True
            else:
                right_distance = np.linalg.norm(
                    points[right_point_index] - center)
                right_is = right_distance > self._max_grasp_width / 2
            # 判断夹爪左边是否不会碰到
            if left_point_index >= points_len or left_point_index < 0:
                left_is = True
            else:
                left_distance = np.linalg.norm(
                    points[left_point_index] - center)
                left_is = left_distance > self._max_grasp_width / 2
            # 如果通过检测
            if distance_is and left_is and right_is:
                grasp = Grasp_2f(center, p-given_point,
                                 self._max_grasp_width, config=self._config)
                # logging.debug('_find_grasp 找到一个合适的抓取，物体:%s ' % (mesh.name))
                grasps.append(grasp)
        if len(grasps) > 0:
            logging.debug('_find_grasp 成功生成%d个抓取，物体:%s ' %
                          (len(grasps), mesh.name))
            return True, grasps
        logging.debug('_find_grasp 抓取生成失败，物体:%s ' % (mesh.name))
        return False, None

    def sample_grasps(self, mesh, num_grasps, num_samples=2):
        """ 采样抓一组取候选点.
        mesh : 待抓取的物体网格
        num_grasps : int采样个数
        num_samples : 每个点采样的方向数
        return: 候选抓取点列表
        """
        grasps = []
        # 获取表面点
        points, face_index = mesh.tri_mesh.sample(
            num_grasps, return_index=True)

        # 对于每个表面点
        for point, face in zip(points, face_index):
            # 计算摩擦锥
            normal = mesh.tri_mesh.face_normals[face]
            c1 = Contact(point, normal, -normal)
            # 在摩擦锥内采样抓取轴
            v_samples = self._sample_vector(c1, num_samples=num_samples)

            for v in v_samples:
                # 找到另一个接触点
                grasp_is, grasp = self._find_grasp(mesh, point, v)

                if not grasp_is:
                    continue

                # 获取真实的接触点 (之前的接触点容易会变化)
                for g in grasp:
                    success, c = g.close_fingers(mesh)
                    if not success:
                        logging.debug('sample_grasps 夹爪闭合失败')
                        continue

                    # 检查摩擦圆锥是否力闭合
                    if force_closure_2f(c[0], c[1], self._friction_coef):
                        grasps.append(g)
                        logging.debug('sample_grasps 成功生成一个抓取')

        # 打乱样本
        random.shuffle(grasps)
        return grasps

    def generate_grasps(self, mesh, target_num_grasps, grasp_gen_mult=2, max_iter=3):
        """ 生成候选抓取点
        mesh : 待抓取的物体
        target_num_grasps : 目标生成抓取点数目
        grasp_gen_mult : 过采样倍数
        max_iter : 最大迭代次数

        Return : 候选抓取列表
        """
        num_grasps_remaining = target_num_grasps

        grasps = []
        k = 1
        while num_grasps_remaining > 0 and k <= max_iter:
            # 过采样抓取点
            num_grasps_generate = grasp_gen_mult * num_grasps_remaining
            new_grasps = self.sample_grasps(mesh, num_grasps_generate)

            # 新加入的夹爪必须和之前的所有夹爪距离大于限定值
            for grasp in new_grasps:
                min_dist = np.inf
                for cur_grasp in grasps:
                    dist = Grasp_2f.distance(cur_grasp, grasp)
                    if dist < min_dist:
                        min_dist = dist
                if min_dist >= self._grasp_dist_thresh:
                    grasps.append(grasp)

            grasp_gen_mult = int(grasp_gen_mult * 2)
            num_grasps_remaining = target_num_grasps - len(grasps)
            k += 1
        if len(grasps) < target_num_grasps:
            logging.warning('generate_grasps 生成数目未达到, 生成数目%d,物体 %s' %
                         (len(grasps), mesh.name))

        random.shuffle(grasps)
        if len(grasps) > target_num_grasps:
            grasps = grasps[:target_num_grasps]
        return grasps
