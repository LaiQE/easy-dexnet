import sys
import os.path
import logging
import trimesh
import pyrender
import numpy as np
from ruamel.yaml import YAML
ROOT_PATH = r'H:\Robot\easy-dexnet'
sys.path.append(os.path.abspath(os.path.join(ROOT_PATH, 'src')))
try:
    import easydexnet as dex
except Exception as e:
    pass


TEST_OBJ_FILE = os.path.join(ROOT_PATH, r'data/bar_clamp.obj')
TEST_LOG_FILE = os.path.join(ROOT_PATH, 'test/test.log')
TEST_CFG_FILE = os.path.join(ROOT_PATH, 'config/test.yaml')


def config_logging(file=None, level=logging.DEBUG):
    """ 配置全局的日志设置 """
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=file, level=level,
                        format=LOG_FORMAT, filemode='w')


def load_config(file):
    """ 加载配置文件 """
    yaml = YAML(typ='safe')   # default, if not specfied, is 'rt' (round-trip)
    with open(file, 'r', encoding="utf-8") as f:
        config = yaml.load(f)
    return config


def main():
    config_logging(TEST_LOG_FILE)
    config = load_config(TEST_CFG_FILE)
    file_name = os.path.splitext(os.path.basename(TEST_OBJ_FILE))[0]
    print('初始配置成功')

    tri_mesh = trimesh.load_mesh(TEST_OBJ_FILE, validate=True)
    mesh = dex.BaseMesh(tri_mesh, name=file_name)
    sampler = dex.GraspSampler_2f(config=config)
    grasps = sampler.generate_grasps(mesh, 25)
    print('夹爪生成成功')
    metrics = config['metrics']
    quality = [dex.grasp_quality(grasp, mesh, metrics) for grasp in grasps]
    quality = (quality - np.min(quality)) / (np.max(quality) - np.min(quality))
    print(quality)
    scene = dex.DexScene(ambient_light=[0.02, 0.02, 0.02],
                         bg_color=[1.0, 1.0, 1.0])
    scene.add_obj(mesh)
    for g,q in zip(grasps, quality):
        if q > 0.5：
            c = q * np.array([255, 0, -255]) + np.array([0, 0, 255])
            c = np.concatenate((c,[255]))
            c = c.astype(int)
            scene.add_grasp(g, color=c)
            scene.add_grasp_center(g)
    pyrender.Viewer(scene, use_raymond_lighting=True)
if __name__ == "__main__":
    
    # print()
    main()
