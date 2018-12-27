import numpy as np


def gauss_elimination_method(A, B):
    """
    高斯消去法,使用部分交换主元法来避免引入舍入误差
    :param A: ndarray方阵,曾广矩阵的参数部分
    :param B: ndarray向量,曾广矩阵的最后一列
    :return: 方程组的解向量
    """
    n = len(A)  # 方程的个数
    for k in np.arange(n - 1):  # 从第一行开始,依次消去每个变量. (最后一行不需要再消了,因为最后一行只剩下一个变量了) [0,n-2]
        # 部分交换主元:当前行(第k行)和第k列中系数绝对值最大的一行交换
        max_row_index = np.argmax(np.abs(A[:, k]))
        if max_row_index is not k:
            A[[k, max_row_index], :] = A[[max_row_index, k], :]  # 交换两行

        for i in np.arange(k + 1, n):  # [k+1,n-1] 从k行下一行开始,直到最后一行
            factor = A[i, k] / A[k, k]
            A[i, :] = A[i, :] - A[k, :] * factor  # 消元
            B[i] = B[i] - B[k] * factor

    # 回代
    X = np.zeros(n)  # 解的向量
    X[n - 1] = B[n - 1] / A[n - 1, n - 1]  # 初始化最后一行确定的变量
    for i in np.arange(n - 2, -1, -1):  # 从倒数第二行开始,一直到最上面
        sum = B[i]
        for j in np.arange(i + 1, n):
            sum = sum - A[i, j] * X[j]  # 第j列(第j个变量)
        X[i] = sum / A[i, i]

    return X


if __name__ == '__main__':
    print('赵励志')
    A = np.array([
        [70, 1, 0],
        [60, -1, -1],
        [40, 0, 1]
    ], dtype=float)
    B = np.array([636, 518, 307], dtype=float)
    X = gauss_elimination_method(A, B)
    print('my solution:', X)
    print('np.linalg.solve solution:', np.linalg.solve(A, B))
