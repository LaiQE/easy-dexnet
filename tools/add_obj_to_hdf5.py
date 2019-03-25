import sys
import os.path
import logging
import glob
import h5py
from ruamel.yaml import YAML

ROOT_PATH = r'H:\Robot\easy-dexnet'
dex_path = os.path.abspath(os.path.join(ROOT_PATH, 'src'))
sys.path.append(dex_path)
try:
    import easydexnet as dex
except Exception as e:
    print('导入模块失败', e)

TEST_CFG_FILE = os.path.join(ROOT_PATH, 'config/add_obj.yaml')

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
    hdf5_path = config['path']['hdf5_path']
    obj_path = config['path']['obj_path']
    config_logging(logging_path, logging.INFO)

    data = h5py.File(hdf5_path, 'a')
    file_list = glob.glob(os.path.join(obj_path, '*.obj'))
    file_list = [os.path.basename(obj) for obj in file_list]
    if config['obj_list'] == 'all':
        obj_list = file_list
    else:
        obj_list = [obj+'.obj' for obj in config['obj_list']
                    if obj+'.obj' in file_list]
    
    if not config['recover']:
        print(config['recover'])
        data_list = list(data.keys())
        obj_list = [obj for obj in obj_list if obj[:-4] not in data_list]
    print(obj_list)

    for obj in obj_list:
        obj_file = os.path.join(obj_path, obj)
        dex_obj = dex.DexObject.from_file(obj_file, config)
        dex_obj.to_hdf5_group(data, config)
    data.close()

if __name__ == "__main__":
    main()