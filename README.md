# easy-dexnet
基于dex-net2.0的论文，较为简单的Dex-Net2.0的实现
对于伯克利的dex-net实现，这里主要进行了如下改动
1. 抓取生成过程中改用tvtk采样表面对映点，而非使用SDF
2. 对于mesh文件的处理全部由trimesh完成
3. 生成图片样本由pyrender库完成
4. 所有程序全部基于python3.5
除了以上的主要改动，由于这里是完全重写的程序，所有大部分的实现细节也都有改动

### 安装部署
easydexnet是基于python3.5编写，需要首先安装python3.5，另外tvtk需要另外安装
> git clone https://github.com/LaiQE/easy-dexnet.git  
cd easy-dexnet  
python setup.py develop

### 使用
- 从obj文件生成所有夹爪与抓取品质并保存到HDF5  
参考tools/add_obj_to_hdf5.py  
需要改动config/add_obj.yaml配置文件
- 从HDF5的数据库中生成gqcnn训练数据  
参考tools/generate.py 
需要改动config/generate.yaml配置文件
- 其他用法  
在easy-dexnet中主要用DexObject类管理所有的数据与计算，其他用法参考DexObject类

### 参考
http://berkeleyautomation.github.io/dex-net/