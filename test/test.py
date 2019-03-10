import sys
import os.path
import logging
import trimesh
from ruamel.yaml import YAML
ROOT_PATH = r'H:\Robot\easy-dexnet'
sys.path.append(os.path.abspath(os.path.join(ROOT_PATH, 'src')))
import easydexnet as dex


TEST_OBJ_FILE = os.path.join(ROOT_PATH, r'data/bar_clamp.obj')
TEST_LOG_FILE = os.path.join(ROOT_PATH, 'test/test.log')
TEST_CFG_FILE = os.path.join(ROOT_PATH, 'config/test.yaml')

def config_logging(file=None, level=logging.DEBUG):
    """ 配置全局的日志设置 """
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=file, level=level, format=LOG_FORMAT, filemode='w')

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

    tri_mesh = trimesh.load_mesh(TEST_OBJ_FILE, validate=True)
    mesh = dex.BaseMesh(tri_mesh, name=file_name)
    sampler = dex.GraspSampler_2f(config=config)
    grasps = sampler.generate_grasps(mesh, 10)
    print(grasps)
    for g in grasps:
        print(g.center, g.axis)

if __name__ == "__main__":
    # print()
    main()
    
