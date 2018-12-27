import numpy as np


def LU_decomposition(A):
    """
    对原始参数矩阵A进行LU分解
    :param A: 原始参数矩阵A
    :return: 分解后的L矩阵,A化为下三角矩阵形式的U矩阵
    """
    n = len(A)
    L = np.eye(n)

    for i in np.arange(n - 1):  # 每一列(去除最后一列)
        L_i = np.eye(n)
        L_i[i + 1:, i] = -A[i + 1:, i] / A[i, i]
        A = np.dot(L_i, A)
        L = np.dot(L, np.linalg.inv(L_i))

    return L, A

def solve(A,B):
    """
    利用LU分解,求解线性方程组
    :param A: 系数矩阵
    :param B: 值向量
    :return: 根向量
    """
    L, U = LU_decomposition(A)
    D = np.linalg.solve(L,B)
    # 回代
    n = len(L)
    X = np.zeros(n)  # 解的向量
    X[n - 1] = D[n - 1] / U[n - 1, n - 1]  # 初始化最后一行确定的变量
    for i in np.arange(n - 2, -1, -1):  # 从倒数第二行开始,一直到最上面
        sum = D[i]
        for j in np.arange(i + 1, n):
            sum = sum - U[i, j] * X[j]  # 第j列(第j个变量)
        X[i] = sum / U[i, i]
    return X

if __name__ == '__main__':
    A = np.array([
        [70, 1, 0],
        [60, -1, -1],
        [40, 0, 1]
    ], dtype=float)
    B = np.array([636, 518, 307], dtype=float)
    X = solve(A,B)
    print('my solution:', X)
    print('np.linalg.solve solution:', np.linalg.solve(A, B))
