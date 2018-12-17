import numpy as np
import sympy
import matplotlib.pyplot as plt


def lagrange_polynomial(X, Y):
    """
    利用拉格朗日插值法拟合数据点
    尽量不用for循环,改为用矩阵加速运算
    :param X: (len,)  ndarray
    :param Y: (len,)  ndarray
    :return: 拟合函数f(x)
    """
    x = sympy.Symbol('x')  # 符号x
    n = len(X)
    # [0,1,2,3,..,n]行向量列向量广播相减后转为01矩阵,为了保证ij不相等
    i = np.arange(n).reshape(n, 1)  # 列向量,0-n
    j = np.arange(n)  # 行向量,0-n
    ij_check = np.array(i - j, dtype=bool).astype(int)  # (n,n)

    xi_sub_xj = X.reshape(n, 1) - X  # (n,n)
    xi_sub_xj[xi_sub_xj == 0] = 1  # 把xi-xj为0 的部分转为1,也就是去除掉ij相等的部分

    x_sub_xj = (x - X) * ij_check  # 符号x减向量X,竖向广播后去除ij相等的部分  (n,)
    x_sub_xj[x_sub_xj == 0] = 1  # 0转1

    L = x_sub_xj / xi_sub_xj  # 拉格朗日多项式
    L = L.prod(axis=1)  # 横向对i连续乘积 (n,)

    fx = (L * Y).sum()  # fx是乘积求和
    return sympy.lambdify(x, fx, 'numpy')  # sympy符号表达式转为函数


if __name__ == '__main__':
    print('赵励志 1607094146')

    # test1-----------------------------------
    X = np.array([1, 4, 6])
    Y = np.array([0, 1.3863, 1.7918])
    f = lagrange_polynomial(X, Y) # 可以拟合任何一组数据
    print(f(2))
    # test1-----------------------------------

    # test2-----------------------------------
    from sklearn import datasets
    iris = datasets.load_iris()

    X = iris.data[:1000,1]
    Y = iris.target[:1000]
    f = lagrange_polynomial(X, Y) # 可以拟合任何一组数据

    plt.plot(np.linspace(2, 5, 100), f(np.linspace(-10, 10, 100)))
    plt.scatter(X, Y)
    plt.show()
    # test2-----------------------------------
