import numpy as np


class Camera(object):
    def __init__(self, pose, model):
        self._pose = pose
        self._model = model

    @property
    def pose(self):
        return self._pose

    @property
    def model(self):
        return self._model


class RandomCamera(Camera):
    """ 随机生成相机参数的类
    """

    def __init__(self, config):
        if 'camera' in config.keys():
            config = config['camera']
        self._config = config
        self._pose = self.random_pose()
        self._model = self.random_model()

    @staticmethod
    def sph2cart(r, az, elev):
        """ 转换球坐标系到笛卡尔坐标系
        r : 半径, 即远点到目标点的距离
        az : 方位角, 绕z轴旋转的角度
        elev : 俯仰角, 与z轴的夹角 
        """
        x = r * np.cos(az) * np.sin(elev)
        y = r * np.sin(az) * np.sin(elev)
        z = r * np.cos(elev)
        return x, y, z

    def random_pose(self, num=1):
        poses = []
        cfg = self._config
        for _ in range(num):
            radius = np.random.uniform(cfg['min_radius'], cfg['max_radius'])
            elev = np.deg2rad(np.random.uniform(
                cfg['min_elev'], cfg['max_elev']))
            az = np.deg2rad(np.random.uniform(cfg['min_az'], cfg['max_az']))
            roll = np.deg2rad(np.random.uniform(
                cfg['min_roll'], cfg['max_roll']))
            x = np.random.uniform(cfg['min_x'], cfg['max_x'])
            y = np.random.uniform(cfg['min_y'], cfg['max_y'])
            poses.append(self._camera_pose(radius, elev, az, roll, x, y))
        return np.squeeze(poses)

    def random_model(self, num=1):
        models = []
        cfg = self._config
        for _ in range(num):
            yfov = np.deg2rad(np.random.uniform(
                cfg['min_yfov'], cfg['max_yfov']))
            znear = np.random.uniform(cfg['min_znear'], cfg['max_znear'])
            aspectRatio = cfg['aspectRatio']
            models.append(np.array([yfov, znear, aspectRatio]))
        return np.squeeze(models)

    def _camera_pose(self, radius, elev, az, roll, x, y):
        """ 从给定的参数计算相机的位置矩阵
        radius : 相机到原点的距离
        elev : 俯仰角, 与z轴的夹角
        az : 方位角, 绕z轴旋转的角度
        roll : 横滚角，绕相机z轴旋转的夹角
        x,y : 物体的x,y轴偏移, 这里折算到相机的位姿上 """
        # 生成相机的中点位置
        delta_t = np.array([x, y, 0])
        camera_center_obj = np.array(
            [self.sph2cart(radius, az, elev)]).squeeze() + delta_t
        camera_z_obj = np.array([self.sph2cart(radius, az, elev)]).squeeze()
        camera_z_obj = camera_z_obj / np.linalg.norm(camera_z_obj)

        # 计算x轴和y轴方向, x轴在水平面上
        camera_x_par_obj = np.array([camera_z_obj[1], -camera_z_obj[0], 0])
        if np.linalg.norm(camera_x_par_obj) == 0:
            camera_x_par_obj = np.array([1, 0, 0])
        camera_x_par_obj = camera_x_par_obj / np.linalg.norm(camera_x_par_obj)
        camera_y_par_obj = np.cross(camera_z_obj, camera_x_par_obj)
        camera_y_par_obj = camera_y_par_obj / np.linalg.norm(camera_y_par_obj)
        # 保证y轴朝下
        if camera_y_par_obj[2] > 0:
            camera_y_par_obj = -camera_y_par_obj
            camera_x_par_obj = -camera_x_par_obj.copy()
        # camera_y_par_obj = -camera_y_par_obj

        # 旋转相机
        R_obj_camera_par = np.c_[camera_x_par_obj,
                                 camera_y_par_obj, camera_z_obj]
        R_camera_par_camera = np.array([[np.cos(roll), -np.sin(roll), 0],
                                        [np.sin(roll), np.cos(roll), 0],
                                        [0, 0, 1]])
        R_obj_camera = R_obj_camera_par.dot(R_camera_par_camera)

        matrix = np.eye(4)
        matrix[:3, :3] = R_obj_camera
        matrix[:3, 3] = camera_center_obj

        return matrix
