import random
import numpy as np
from .grasp import Grasp_2f
from .contact import Contact


class GraspSampler_2f(object):
    def __init__(self, width, min_contact_dist):
        self._max_grasp_width = width
        self._min_contact_dist = min_contact_dist

    def _sample_vector(self, contact, friction_coef=0.5, num_samples=1):
        """ 在给定点接触点的摩擦锥内采样随机的方向向量
        采样的方向随机向内或向外,这里朝向无所谓
        contract : 一个接触点
        friction_coef : 摩擦系数
        num_samples : 采样个数
        return : 方向向量列表
       """
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
        if points_len % 2 != 0:
            return False, None

        # 找到当前点的位置, 由于计算误差用一个极小数测试是否一致
        for i, p in enumerate(points):
            if np.linalg.norm(p - given_point) < 1.e-6:
                given_point_index = i
                break
        # 如果给定点没有在计算出的交点内，则出错
        if given_point_index == -1:
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
        # 对于每一个候选配对点检查距离是否小于最小距离或是否会被下一个点挡住
        for inedx in candidate_point_index:
            p = points[inedx]
            # 判断两个点之间的距离是否大于要求的最小距离
            distance = np.linalg.norm(given_point - p)
            distance_is = distance > self._min_contact_dist

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
                grasp = Grasp_2f(center, p-given_point, self._max_grasp_width)
                return True, grasp
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
            # 计算摩擦圆锥面
            normal = mesh.tri_mesh.face_normals[face]
            c1 = Contact(point, normal, -normal)
            # 在摩擦圆锥内采样抓取轴
            v_samples = self._sample_vector(c1, num_samples=num_samples)

            for v in v_samples:
                # 找到另一个接触点
                grasp_is, grasp = self._find_grasp(mesh, point, v)

                if not grasp_is:
                    continue

                # 获取真实的接触点 (之前的接触点容易会变化)
                success, c0, c1 = grasp.close_fingers(mesh)
                if not success:
                    continue

                # TODO 检查摩擦圆锥是否力闭合
                # if PointGraspMetrics3D.force_closure(c1, c2, self.friction_coef):
                #     grasps.append(grasp)
                grasps.append(grasp)

        # randomly sample max num grasps from total list
        random.shuffle(grasps)
        return grasps
