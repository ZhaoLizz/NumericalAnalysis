import numpy as np
import matplotlib.pyplot as plt

def heun(equation,step,x0,y0,X):
    """
    利用修恩法求解微分方程
    :param equation: 线性微分方程 np.poly1d
    :param step: 步长
    :param x: 初始条件x0
    :param y: 初始条件y0
    :param X: 求解出的原始方程的自变量
    :return:
    """
    x_now = x0
    y_now = y0

    while np.where(x0 < X,x_now < X,x_now >= X):
        k = (equation(x_now) + equation(x_now + step)) / 2
        step = step if X - x_now > 0 else -step # 判断step的方向
        y_now = y_now + k * step
        x_now = x_now + step
    return y_now


if __name__ == '__main__':
    dydx = np.poly1d([-2,12,20,8.5])
    x0,y0 = 0,1
    step = 0.5

    y = heun(dydx,step,0,1,0.5)
    print(y)

    plt.plot(np.linspace(-2,8.5,100),dydx(np.linspace(-2,8.5,100)))
    plt.show()

