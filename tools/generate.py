import sys
import os.path
import logging
import h5py
from ruamel.yaml import YAML

ROOT_PATH = r'H:\Robot\easy-dexnet'
dex_path = os.path.abspath(os.path.join(ROOT_PATH, 'src'))
sys.path.append(dex_path)
try:
    import easydexnet as dex
except Exception as e:
    print('导入模块失败', e)

TEST_CFG_FILE = os.path.join(ROOT_PATH, 'config/generate.yaml')
OBJ_GROUP = 'datasets/mini_dexnet/objects/'
DEX_DATA = False



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
    config = load_config(TEST_CFG_FILE)
    logging_path = config['path']['logging_path']
    data_path = config['path']['data_path']
    out_path = config['path']['out_path']
    table_path = config['path']['table_path']
    config_logging(logging_path, logging.INFO)

    table = dex.BaseMesh.from_file(table_path)
    saver = dex.DexSaver(out_path, config)

    data = h5py.File(data_path, 'a')
    if DEX_DATA:
        data = data[OBJ_GROUP]
    if config['obj_list'] == 'all':
        obj_list = data.keys()
    else:
        obj_list = [obj for obj in config['obj_list']
                    if obj in list(data.keys())]
    for obj_name in obj_list:
        obj_group = data[obj_name]
        dex_obj = dex.DexObject.from_hdf5_group(obj_group, config, obj_name)
        depth_render = dex.DepthRender(dex_obj, table, saver, config)
        depth_render.render()
    saver.close()
    data.close()


if __name__ == "__main__":
    main()
