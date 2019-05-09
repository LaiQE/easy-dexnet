import numpy as np


class Grasp2D(object):
    """
    2D夹爪类型，夹爪投影到深度图像上的坐标.
    center : 夹爪中心坐标，像素坐标表示
    angle : 抓取方向和相机x坐标的夹角
    depth : 夹爪中心点的深度
    width : 夹爪的宽度像素坐标
    """

    def __init__(self, center, angle, depth, width=0.0):
        self._center = center
        self._angle = angle
        self._depth = depth
        self._width_px = width

    @property
    def center(self):
        return self._center.astype(np.int)

    @property
    def center_float(self):
        return self._center

    @property
    def angle(self):
        return self._angle

    @property
    def depth(self):
        return self._depth

    @property
    def axis(self):
        """ Returns the grasp axis. """
        return np.array([np.cos(self._angle), np.sin(self._angle)])

    @property
    def endpoints(self):
        """ Returns the grasp endpoints """
        p0 = self._center - (float(self._width_px) / 2) * self.axis
        p1 = self._center + (float(self._width_px) / 2) * self.axis
        p0 = p0.astype(np.int)
        p1 = p1.astype(np.int)
        return p0, p1

    @property
    def feature_vec(self):
        """ 生成抓取的特征向量，由两个端点和中心距离组成
        """
        p1, p2 = self.endpoints
        return np.r_[p1, p2, self._depth]

    @staticmethod
    def from_feature_vec(v, width=None):
        """ 通过特征向量创建一个抓取-
        v : 抓取的特征向量
        width : 夹爪的宽度
        """
        # read feature vec
        p1 = v[:2]
        p2 = v[2:4]
        depth = v[4]
        if width is None:
            width = np.linalg.norm(p1 - p2)

        # compute center and angle
        center_px = (p1 + p2) / 2
        axis = p2 - p1
        if np.linalg.norm(axis) > 0:
            axis = axis / np.linalg.norm(axis)
        if axis[1] > 0:
            angle = np.arccos(axis[0])
        else:
            angle = -np.arccos(axis[0])
        return Grasp2D(center_px, angle, depth, width=width)

    @staticmethod
    def image_dist(g1, g2, alpha=1.0):
        """ 计算两个抓取在像素坐标下的距离
        """
        # point to point distances
        point_dist = np.linalg.norm(g1.center - g2.center)

        # axis distances
        axis_dist = np.arccos(np.abs(g1.axis.dot(g2.axis)))

        return point_dist + alpha * axis_dist

    def to_saver(self):
        s = np.zeros((5,))
        s[:2] = self._center
        s[2] = self._depth
        s[3] = self._angle
        s[4] = self._width_px
        return s
