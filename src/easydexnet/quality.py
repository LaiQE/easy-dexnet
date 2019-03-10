import numpy as np
from .contact import Contact

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