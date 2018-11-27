def test_position_method(fun, a, b, epsilon):
    """
    试位法寻找函数fun在区间[a,b]近似根
    :param fun: 目标函数
    :param a: 区间左下标
    :param b: 区间右下标
    :param epsilon: 误差要求,以区间长度b-a来度量
    :return: 函数fun在区间[a,b]近似根
    """
    if fun(a) * fun(b) > 0:
        raise Exception('there is no root in section [a,b]')
    while b - a > epsilon:
        f_a ,f_b = fun(a),fun(b)
        x_mid = (abs(f_a ) * b + abs(f_b) * a ) / (abs(f_b) + abs(f_a))
        f_xmid = fun(x_mid)
        if f_xmid * f_a < 0:
            b = x_mid
        else:
            a = x_mid
    return (abs(fun(a) ) * b + abs(fun(b)) * a ) / (abs(fun(b)) + abs(fun(a)))


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
    x = test_position_method(fun, a, b, epsilon)
    print(x)

    print(fun(x))
