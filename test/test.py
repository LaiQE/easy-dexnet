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

def plot_grasp2d(grasp_2d, size):
    def plot_2p(p0, p1):
        x = [p0[0], p1[0]]
        y = [p0[1], p1[1]]
        plt.plot(x,y)
    p0,p1 = grasp_2d.endpoints
    # axis = grasp_2d.axis
    # axis_T = np.array([-axis[1], axis[0]])
    # axis_T = axis_T / np.linalg.norm(axis_T)
    # p00 = p0 + size * axis_T
    # p00 = p00.astype(np.int)
    # p01 = p0 - size * axis_T
    # p01 = p01.astype(np.int)
    # p10 = p1 + size * axis_T
    # p10 = p10.astype(np.int)
    # p11 = p1 - size * axis_T
    # p11 = p11.astype(np.int)
    plot_2p(p0, p1)
    # plot_2p(p00, p01)
    # plot_2p(p10, p11)

def plot_2p(p0, p1):
    x = [p0[0], p1[0]]
    y = [p0[1], p1[1]]
    plt.plot(x,y)


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
    grasps = dex.DexObject.get_grasps(mesh, config)
    
    pose = poses[0]

    vaild_grasps = []
    for g in grasps:
        if g.check_approach(mesh, pose, config) and \
            g.get_approch(pose)[1] < 40:
            vaild_grasps.append(g)
    
    # vaild_grasps_T = [g.apply_transform(pose.matrix) for g in vaild_grasps]
    # mesh_T = mesh.apply_transform(pose.matrix)
    # display(mesh_T, vaild_grasps_T)
    
    render = dex.ImageRender(mesh, pose, table, config)
    # for g in vaild_grasps:
    #     render.scene.add_grasp(g, render._obj_matrix)
    # pyrender.Viewer(render.scene, use_raymond_lighting=True)
    rgb, depth = render.data
    # plt.imshow(rgb)
    # for g in vaild_grasps:
    #     plot_2p(*render.render_grasp(g).endpoints)
    # plt.show()
    # display(mesh_T, vaild_grasps_T)
    for g in vaild_grasps:
        g_2d = render.render_grasp(g)
        out = dex.DataGenerator(rgb, g_2d, config).output
        plt.imshow(out)
        plot_2p([9,16], [23,16])
        plt.show()

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.axis('off')
    # plt.imshow(rgb)
    # plt.subplot(1,2,2)
    # plt.axis('off')
    # plt.imshow(depth, cmap=plt.cm.gray_r)
    # plt.imshow(depth)
    # plt.colorbar()
    # plt.show()
    
if __name__ == "__main__":
    
    # print()
    test_render()
