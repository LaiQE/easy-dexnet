import sys
import os.path
import logging
import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
ROOT_PATH = r'H:\Robot\easy-dexnet'
sys.path.append(os.path.abspath(os.path.join(ROOT_PATH, 'src')))
try:
    import easydexnet as dex
except Exception as e:
    pass

# 测试一下github的同步功能，这段文字来自于ubuntu


TEST_OBJ_FILE = os.path.join(ROOT_PATH, r'data/bar_clamp.obj')
TEST_LOG_FILE = os.path.join(ROOT_PATH, 'test/test.log')
TEST_CFG_FILE = os.path.join(ROOT_PATH, 'config/test.yaml')
TEST_TABLE_FILE = os.path.join(ROOT_PATH, r'data/table.obj')


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

def display(mesh, grasps, quality_s=None):
    scene = dex.DexScene(ambient_light=[0.02, 0.02, 0.02],
                         bg_color=[1.0, 1.0, 1.0])
    scene.add_obj(mesh)
    if quality_s is None:
        quality_s = [1] * len(grasps)
    for g,q in zip(grasps, quality_s):
        c = q * np.array([255, 0, -255]) + np.array([0, 0, 255])
        c = np.concatenate((c,[255]))
        c = c.astype(int)
        scene.add_grasp(g, color=c)
        scene.add_grasp_center(g)
    pyrender.Viewer(scene, use_raymond_lighting=True)


def main():
    config_logging(TEST_LOG_FILE)
    config = load_config(TEST_CFG_FILE)

    dex_obj = dex.DexObject.from_file(TEST_OBJ_FILE, config)
    quality = dex_obj.qualitis
    quality_s = (quality - np.min(quality)) / (np.max(quality) - np.min(quality))
    display(dex_obj.mesh, dex_obj.grasps, quality_s)
    pose = dex_obj.poses[0]
    grasps = []
    quality = []
    for g,q in zip(dex_obj.grasps,quality_s):
        if g.check_approach(dex_obj.mesh, pose, config) and \
            g.get_approch(pose)[1] < 40:
            grasps.append(g.apply_transform(pose.matrix))
            quality.append(q)
    mesh = dex_obj.mesh.apply_transform(pose.matrix)
    display(mesh, grasps, quality)

def test_render():
    config_logging(TEST_LOG_FILE)
    config = load_config(TEST_CFG_FILE)

    mesh = dex.BaseMesh.from_file(TEST_OBJ_FILE)
    table = dex.BaseMesh.from_file(TEST_TABLE_FILE)
    poses = dex.DexObject.get_poses(mesh, 0.0)
    render = dex.ImageRender(mesh, poses[0], table, config)
    m = mesh.center_mass
    print(render.render_obj_point(m))
    # _, depth = render.data
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.axis('off')
    # plt.imshow(image)
    # plt.subplot(1,2,2)
    # plt.axis('off')
    # plt.imshow(depth, cmap=plt.cm.gray_r)
    # plt.imshow(depth)
    # plt.colorbar()
    # plt.show()
    
if __name__ == "__main__":
    
    # print()
    test_render()
