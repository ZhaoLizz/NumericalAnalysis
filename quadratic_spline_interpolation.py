import numpy as np
import matplotlib.pyplot as plt


def quadratic_spline(X, Y):
    """
    二次样条插值法
    具有泛化性,可以用二次样条插值处理任意的数据
    利用下标的递推特点构造线性方程.时间复杂度O(n)
    :param X:
    :param Y:
    :return: 样条函数
    """
    n = len(X) - 1  # n+1个点,n个区间

    A = np.zeros((3 * n, 3 * n))
    B = np.zeros(3 * n)

    for i in range(n):  # [0,n-1]
        # ai * xi**2 + bi * xi + ci = f(xi)
        A[i, i * 3] = X[i] ** 2  # ai
        A[i, i * 3 + 1] = X[i]  # bi
        A[i, i * 3 + 2] = 1  # ci
        B[i] = Y[i]

        # ai * x_(i+1)**2 + bi * x_(i+1) + ci = f(x(i+1))
        A[n + i, i * 3] = X[i + 1] ** 2  # ai
        A[n + i, i * 3 + 1] = X[i + 1]  # bi
        A[n + i, i * 3 + 2] = 1  # ci
        B[n + i] = Y[i + 1]

        # 一阶导相等
        if i <= n - 2:
            A[2 * n + i, i * 3] = 2 * X[i + 1]
            A[2 * n + i, i * 3 + 1] = 1
            A[2 * n + i, (i + 1) * 3] = -2 * X[i + 1]
            A[2 * n + i, (i + 1) * 3 + 1] = -1

    # 限制a0 = 0
    A[3 * n - 1, 0] = 1

    return np.linalg.solve(A, B)


def function(parameters, origin_X, X):
    """
    分段函数
    :param parameters: 二次样条插值函数的参数
    :param origin_X: 数据集的X,ndarray
    :param X: 要预测的X
    :return:
    """
    X = np.array(X)

    index = np.argmax(X.reshape(len(X), 1) < origin_X, axis=1) - 2  # X所处的区间. index-1 到 index. index指示abc_i
    print(np.argmax(X.reshape(len(X), 1) < origin_X, axis=1))
    print(index)

    index[index < 0] = 0

    print(index)
    # return lambda x :  parameters[index*3] * x **2 + parameters[index * 3 + 1] * x + parameters[index * 3 + 2]
    return parameters[index * 3] * X ** 2 + parameters[index * 3 + 1] * X + parameters[index * 3 + 2]


if __name__ == '__main__':
    X = [3, 4.5, 7, 9]
    Y = [2.5, 1, 2.5, 0.5]
    parameters = quadratic_spline(X, Y)
    print(parameters)

    X_pred = np.linspace(4, 8, 100)
    y_pred = function(parameters, X, X_pred)

    # f = function(parameters,X,X_pred)
    plt.plot(X_pred, y_pred)
    plt.scatter(np.array(X), np.array(Y))
    plt.show()
