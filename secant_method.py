import sympy
import numpy as np


def secant_method(f, epsilon, x_0):
    """
    利用弦截法迭代逼近方程的根
    :param f: 方程 f(x) = 0,lambda类型
    :param epsilon: 最大误差
    :param x_0: 迭代起点
    :return:
    """
    x_1 = 1
    x_list = [x_0, x_1]

    i = 1
    while abs(x_list[i] - x_list[i - 1]) > epsilon:
        x_new = x_list[i] - (f(x_list[i]) * (x_list[i] - x_list[i - 1])) / (f(x_list[i]) - f(x_list[i - 1]))
        x_list.append(x_new)
        i += 1

    return x_list


f = lambda x: x ** 3 - x - 1
epsilon = 0.0001
x_0 = 1.5
x_list = secant_method(f, epsilon, x_0)
print(x_list[-1])
