import sympy
import numpy as np

def newton_method(f, epsilon, x_0):
    """
    利用牛顿法迭代逼近方程的根
    :param f: 方程 f(x) = 0 ,sympy表达式类型
    :param epsilon: 最大误差
    :param x_0: 迭代起点
    :return:
    """
    f_diff = sympy.diff(f,'x') # 对x的一阶导数
    f_iter = x - f / f_diff # 迭代函数
    f_iter_diff = sympy.diff(f_iter,'x') # 迭代函数一阶导数
    if abs(f_iter_diff.evalf(subs={'x':x_0})) > 1:
        raise Exception('iter function does not converge')

    i = 1
    x_next = f_iter.evalf(subs={'x':x_0}) # x_(k+1)
    x_list = [x_0,x_next]

    # 利用迭代函数迭代,记录每次的x点
    while abs(x_list[i] - x_list[i-1]) > epsilon:
        i += 1
        x_next = f_iter.evalf(subs={'x': x_list[i-1]})
        x_list.append(x_next)
    return np.array(x_list,dtype='float')


x = sympy.Symbol('x')
f = x**3 - x - 1
epsilon = 0.0001
x_0 = 1.5
x_list = newton_method(f,epsilon,x_0)
print(x_list[-1])
