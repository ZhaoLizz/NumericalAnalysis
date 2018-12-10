import matplotlib.pyplot as plt
import numpy as np


def difference_quotient(i, j, X, Y):
    """
    i到j的n阶有限差商 (i > j)
    i,j是已知点的起点和终点下标
    n = i - j
    :param i:x_i对应的下标
    :param j:
    :return:
    """
    if i - j == 1:
        return (Y[i] - Y[j]) / (X[i] - X[j])
    return ((difference_quotient(i, j + 1, X, Y) - difference_quotient(i - 1, j, X, Y)) / (X[i] - X[j]))


if __name__ == '__main__':
    print('赵励志')
    X = np.array([1, 3, 2])
    Y = np.array([1, 2, -1])
    b = [Y[0]]

    n = len(X) - 1  # 最高n阶
    for i in range(1, n + 1):  # 1到n阶
        b.append(difference_quotient(i, 0, X, Y))

    # 这里牛顿多项式表示形式本来想优化一下泛化能力,写成函数的形式. 可惜没时间了  赶着交作业,暂时先写成固定形式吧.
    f = lambda x: b[0] + b[1] * (x - X[0]) + b[2] * (x - X[0]) * (x - X[1])
    print('b', b)
    plt.plot(np.linspace(0, 5, 100), f(np.linspace(0, 5, 100)))
    plt.scatter(X, Y, marker='o')
    plt.show()
