import sympy
import numpy as np

def base_iterative_method(f_iter,epsilon,x_0):
    """
    利用不动迭代法逼近方程的根
    :param f_iter: 迭代公式
    :param epsilon: 误差范围
    :param x_0: x*附近的根
    :return: 每次迭代计算获取的x值
    """
    # 首先判断迭代函数的收敛性,即 一阶导数在x_0处的值的绝对值<1
    f_diff = sympy.diff(f_iter,sympy.Symbol('x'))
    if abs(f_diff.evalf(subs={'x':x_0})) > 1 :
        raise Exception('iter function does not converge')


    i = 1 # 迭代次数,初始化为1
    x_next = f_iter.evalf(subs={'x':x_0}) # x_(k+1)
    x_list = [x_0,x_next] # 每次迭代找到的x值

    while abs(x_list[i] - x_list[i-1]) > epsilon:
        i += 1
        x_next = f_iter.evalf(subs={'x': x_list[i-1]})
        x_list.append(x_next)
        print(x_next)

    return np.array(x_list)



x = sympy.Symbol('x')
f_iter = (x + 1) ** sympy.Rational(1,3)
epsilon = 0.0001
x_0 = 1.5
x_list = base_iterative_method(f_iter,epsilon,x_0)