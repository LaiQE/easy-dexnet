import logging
import numpy as np
from .contact import Contact
from .quality_function import PointGraspMetrics3D

def force_closure_2f(c1, c2, friction_coef, use_abs_value=False):
    """" 检查两个接触点是否力闭合
    c1 : 第一个接触点
    c2 : 第二个接触点
    friction_coef : 摩擦系数
    use_abs_value : 当抓取点的朝向未指定时，这个参数有用
    Returns 0，1表示是否力闭合
    """
    if c1.point is None or c2.point is None or c1.normal is None or c2.normal is None:
        return 0
    p1, p2 = c1.point, c2.point
    n1, n2 = -c1.normal, -c2.normal # inward facing normals

    if (p1 == p2).all(): # same point
        return 0

    for normal, contact, other_contact in [(n1, p1, p2), (n2, p2, p1)]:
        diff = other_contact - contact
        normal_proj = normal.dot(diff) / np.linalg.norm(normal)
        if use_abs_value:
            normal_proj = abs(normal_proj)

        if normal_proj < 0:
            return 0 # wrong side
        alpha = np.arccos(normal_proj / np.linalg.norm(diff))
        if alpha > np.arctan(friction_coef):
            return 0 # outside of friction cone
    return 1

def grasp_quality(grasp, obj, params):
    """
    计算抓取品质, 在Dex-Net中用到了机器学习方法产生的鲁棒抓取品质
    这里为了简单起见, 只是简单的使用epsilon品质
    grasp : 要计算质量的抓取
    obj : 计算抓取的物体
    params : 抓取参数, 参数列表里面的metrics
    target_wrench, 这个参数与稳定姿态有关, 参数表里并未设置
    """
    method = params['quality_method']
    friction_coef = params['friction_coef']
    num_cone_faces = params['num_cone_faces']
    soft_fingers = params['soft_fingers']
    if not hasattr(PointGraspMetrics3D, method):
        raise ValueError('Illegal point grasp metric %s specified' %(method))

    contacts_found, contacts = grasp.close_fingers(obj)
    
    if not contacts_found:
        logging.error('抓取品质计算, 接触点无法找到')
        return 0
    
    if method == 'force_closure':
        if len(contacts) == 2:
            c1, c2 = contacts
            return PointGraspMetrics3D.force_closure(c1, c2, friction_coef)
        method = 'force_closure_qp'

    num_contacts = len(contacts)
    forces = np.zeros([3,0])
    torques = np.zeros([3,0])
    normals = np.zeros([3,0])
    for i in range(num_contacts):
        contact = contacts[i]
        # 计算摩擦锥
        force_success, contact_forces, contact_outward_normal = contact.friction_cone(num_cone_faces, friction_coef)

        if not force_success:
            logging.error('抓取品质计算, 摩擦锥计算失败')
            return 0
            
        # 计算摩擦力矩
        torque_success, contact_torques = contact.torques(contact_forces)
        if not torque_success:
            logging.error('抓取品质计算, 摩擦力矩计算失败')
            return 0

        # 计算法向力大小
        n = contact.normal_force_magnitude()

        forces = np.c_[forces, n * contact_forces]
        torques = np.c_[torques, n * contact_torques]
        normals = np.c_[normals, n * -contact_outward_normal] # store inward pointing normals


    if normals.shape[1] == 0:
        logging.error('抓取品质计算, 法线计算失败')
        return 0

    # normalize torques
    if 'torque_scaling' not in params.keys():
        torque_scaling = 1.0
        if method == 'ferrari_canny_L1':
            _, mx = obj.bounding_box()
            torque_scaling = 1.0 / np.median(mx)
        params['torque_scaling'] = torque_scaling
        
    Q_func = getattr(PointGraspMetrics3D, method)
    quality = Q_func(forces, torques, normals,
                        soft_fingers=soft_fingers,
                        params=params)

    return quality