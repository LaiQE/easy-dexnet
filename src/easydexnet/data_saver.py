import logging
import os
import numpy as np


class DataSaver(object):
    def __init__(self, path, max_data, name):
        self._path = path
        self._max = max_data
        self._counter = 0
        self._buffer = None
        self._save_counter = 0
        self._name = name

    def update_counter(self):
        self._counter = self._counter + 1
        if self._counter == self._max:
            self._counter = 0
            self.save()

    def add(self, data):
        if self._buffer is None:
            self._buffer = np.zeros((self._max,) + data.shape)
        if not(data.shape == self._buffer.shape[1:]):
            raise ValueError()
        self._buffer[self._counter] = data
        self.update_counter()

    def save(self):
        save_name = self._name + '%06d' % (self._save_counter) + '.npy'
        self._save_counter = self._save_counter + 1
        save_file = os.path.join(self._path, save_name)
        np.save(save_file, self._buffer)

    def close(self):
        if self._counter > 0:
            self._buffer = self._buffer[:self._counter]
            self.save()
        total_num = self._counter + (self._save_counter - 1) * self._max
        logging.info(self._name+' saver: 共存储了%d个数据' % (total_num))


class DexSaver(object):
    def __init__(self, path, config):
        self._max = config['datapoints_per_file']
        self._path = path
        self._depth_saver = self.create_saver('depth')
        self._hand_pose_saver = self.create_saver('hand_pose')
        self._quality_saver = self.create_saver('quality')

    def create_saver(self, name):
        path = os.path.join(self._path, name)
        if not os.path.exists(path):
            os.mkdir(path)
        return DataSaver(path, self._max, name)

    def add(self, depth, grasp, q, coll):
        self._depth_saver.add(depth)
        self._hand_pose_saver.add(grasp.to_saver())
        coll_free_metric = np.array([(1 * coll) * q])
        self._quality_saver.add(coll_free_metric)

    def close(self):
        self._depth_saver.close()
        self._hand_pose_saver.close()
        self._quality_saver.close()
