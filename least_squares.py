import numpy as np
import matplotlib.pyplot as plt

def least_squares(X, Y):
    """
    最小二乘直线拟合
    :param X: 向量,ndarray
    :param Y: 向量,ndarray
    :return: 拟合直线的参数向量 (y = a + bx)
    """
    m = len(X)

    A = np.array([
        [m, X.sum()],
        [X.sum(),np.sum(X**2)]
    ])
    b = np.array([np.sum(Y),np.sum(X * Y)])

    return np.linalg.solve(A,b)




if __name__ == '__main__':
    X = np.array([0,0.2,0.4,0.6,0.8])
    Y = np.array([0.9,1.9,2.8,3.3,4.2])
    a = least_squares(X,Y)

    # 绘图
    x = np.linspace(0, 2, 100)
    plt.plot(x,a[0] + a[1] * x,label='linear')
    plt.scatter(X,Y,marker='o')
    plt.legend()
    plt.show()
    print(a)



