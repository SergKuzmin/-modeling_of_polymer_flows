import numpy as np
from numpy.linalg import inv
from test.utils import time_count


@time_count
def tma_solver(a: np.array, b: np.array, c: np.array, f: np.array) -> np.array:
    """
    a[i, :, :] * y[i + 1, :] - b[i, :, :] * y[i, :] + c[i, :, :] * y[i - 1, :] = - f[i, :]
    """
    n_size = b.shape[0]
    shape_of_xz: tuple = (n_size + 1, b.shape[1], b.shape[2])
    x = np.zeros(shape_of_xz)
    z = np.zeros(shape_of_xz[:2])
    y = np.zeros((n_size, b.shape[1]))
    x[1, :, :] = (inv(b[0, :, :])).dot(a[0, :, :])
    z[1, :] = (inv(b[0, :, :])).dot(f[0, :])
    for j in range(1, n_size):
        base: np.array = inv((b[j, :, :] - (c[j, :, :].dot(x[j, :, :]))))
        x[j + 1, :, :] = base.dot(a[j, :, :])
        z[j + 1, :] = base.dot(((c[j, :, :].dot(z[j, :])) + f[j, :]))

    y[n_size - 1, :] = inv((b[n_size - 1, :, :] - (c[n_size - 1, :, :].dot(x[n_size - 1, :, :])))).\
        dot(((c[n_size - 1, :, :].dot(z[n_size - 1, :])) + f[n_size - 1, :]))
    for j in range(n_size - 2, -1, -1):
        y[j, :] = x[j + 1, :, :].dot(y[j + 1, :]) + z[j + 1, :]

    return y


if __name__ == "__main__":
    n = 1000
    k = 5
    a = np.zeros((n, k, k))
    b = np.zeros((n, k, k))
    c = np.zeros((n, k, k))
    f = np.zeros((n, k))
    for i in range(n):
        a[i, :, :] = np.diag([1]*k)
        b[i, :, :] = - np.diag([5]*k)
        c[i, :, :] = np.diag([1]*k)
        if i == 0 or i == n - 1:
            f[i, :] = - np.array([3]*k)
        else:
            f[i, :] = - np.array([4]*k)

    y = tma_solver(a, b, c, f)
    for i in range(n):

        if i == 0:
            print(-b[i, :, :].dot(y[i, :]) + c[i, :, :].dot(y[i+1, :]))
        elif i == n - 1:
            print(a[i, :, :].dot(y[i-1, :]) - b[i, :, :].dot(y[i, :]))
        else:
            print(a[i, :, :].dot(y[i - 1, :]) - b[i, :, :].dot(y[i, :]) + c[i, :, :].dot(y[i + 1, :]))
