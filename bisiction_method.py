import numpy as np


def bisection_method(fun, a, b, epsilon):
    """
    区间二分法寻找函数fun在区间[a,b]近似根
    :param fun: 目标函数
    :param a: 区间左下标
    :param b: 区间右下标
    :param epsilon: 误差要求,以区间长度b-a来度量
    :return: 函数fun在区间[a,b]近似根
    """
    if fun(a) * fun(b) > 0:  # 区间[a,b]内没有根
        raise Exception('there is no root in section [a,b]')

    while b - a > epsilon:
        print('error:', b - a)
        fun_mid = fun((a + b) / 2)
        # np.where()
        # 如果f(mid)和f(a)异号,根位于 [a,mid],调整b
        if fun(a) * fun_mid < 0:
            b = (a + b) / 2
        else:
            a = (a + b) / 2

    return (a + b) / 2


def fun(x):
    """
    函数
    :param x:
    :return: f(x)
    """
    return x ** 3 - x - 1


if __name__ == '__main__':
    a, b = 1, 1.5
    epsilon = 0.00001
    x = bisection_method(fun, a, b, epsilon)
    print(x)

    print(fun(x))
