import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Dict


class Constant:
    def __init__(self):
        self.A_0 = 0.01  # перепад давления в канале
        self.Re = 1  # число Рейнольдса
        self.W = 0.01  # число Вейсенберга
        self.betta = 0.4  # феноменологический параметр
        self.k = 0.6  # феноменологический параметр
        self.k_top_underline = self.k - self.betta  # феноменологический параметр
        self.hi_0_with_plus = 20.8*(10**(-8))
        self.hi_0_with_minus = self.hi_0_with_plus
        self.sigma_m = 1
        self.Pr = 50  # число Прандтля
        self.b_m = 1

        self.A_r = 1
        self.A_m = 1
        self.A_r_top_wavy_underline = 1
        self.A_m_top_wavy_underline = 1
        self.E_A_top_underline = 10
        self.epsilon = 10 ** -6  # коэффициент установления
        # self.t = 0.001  # шаг по времени
        # self.h = 0.001  # шаг по пространству
        self.theta_top_underline = 1  # разность температур на верхней и нижней стенках
        self.J_plus = 2
        self.J_minus = 1
        self.m = 1
        self.k_cap = self.k_top_underline + 3 * self.betta

    @staticmethod
    def big_a(time):
        return 0.001


class Iteration:
    def __init__(self, num: int, time_mult: float):
        self.num = num
        self.c = Constant()
        self.A = np.zeros((num, 4, 4)).astype(np.float64)
        self.B = np.zeros((num, 4, 4)).astype(np.float64)
        self.C = np.zeros((num, 4, 4)).astype(np.float64)
        self.F = np.zeros((num, 4)).astype(np.float64)

        self.X = np.zeros((num, 4, 4)).astype(np.float64)
        self.Y = np.zeros((num, 4)).astype(np.float64)

        self.X_ = np.zeros((num, 4, 4)).astype(np.float64)
        self.Y_ = np.zeros((num, 4)).astype(np.float64)

        self.u = {'n': np.zeros((num,)).astype(np.float64),
                  'n+1': np.zeros((num,)).astype(np.float64) * 0.01,
                  'n+l': np.zeros((num,)).astype(np.float64) * 0.01,
                  'n+l+1': np.zeros((num,)).astype(np.float64) * 0.01}

        self.a_1_1 = {'n': np.zeros((num,)).astype(np.float64) * 0.01,
                      'n+l': np.zeros((num,)).astype(np.float64) * 0.01,
                      'n+l+1': np.zeros((num,)).astype(np.float64) * 0.01}

        self.a_1_2 = {'n': np.zeros((num,)).astype(np.float64) * 0.01,
                      'n+l': np.zeros((num,)).astype(np.float64) * 0.01,
                      'n+l+1': np.zeros((num,)).astype(np.float64) * 0.01,
                      '(n+l)': np.zeros((num,)).astype(np.float64) * 0.01}

        self.a_2_2 = {'n': np.zeros((num,)).astype(np.float64) * 0.01,
                      'n+l': np.zeros((num,)).astype(np.float64) * 0.01,
                      'n+l+1': np.zeros((num,)).astype(np.float64) * 0.01,
                      '(n+l)': np.zeros((num,)).astype(np.float64) * 0.01}

        self.Q = {'n': np.zeros((num,)).astype(np.float64) * 0.01,
                  'n+l': np.zeros((num,)).astype(np.float64) * 0.01,
                  'n+l+1': np.zeros((num,)).astype(np.float64) * 0.01}

        self.R = {'n': np.zeros((num,)).astype(np.float64) * 1,
                  'n+l': np.zeros((num,)).astype(np.float64) * 1,
                  'n+l+1': np.zeros((num,)).astype(np.float64) * 1}

        self.y = np.linspace(-0.5, 0.5, num)
        # self.y = np.linspace(0, 1, num)
        self.h = self.y[1] - self.y[0]
        self.tau = time_mult*(self.y[1] - self.y[0])
        self.r = self.tau / self.h
        # print(self.r)
        self.n = 0  # номер итерации
        self.l = 0  # номер итерации

        self.Z = {'n': self.big_z('n'),
                  'n+l': self.big_z('n+l'),
                  'n+l+1': self.big_z('n+l+1'),
                  '(n+l)': (self.big_z('n+l') + self.big_z('n')) / 2}

        self.L = {'n': self.big_l('n'),
                  'n+l': self.big_l('n+l'),
                  'n+l+1': self.big_l('n+l+1'),
                  '(n+l)': (self.big_l('n+l') + self.big_l('n')) / 2}


        self.sigma = {'n': np.zeros((num,)).astype(np.float64)}
        self.sigma['n+l+1'] = self.small_sigma()

    def big_z(self, index: str, show: bool = True) -> np.array:
        result = self.Q[index] + 1 + self.c.theta_top_underline * (0.5 - self.y)
        # print(result[0], result[-1])
        if show:
            plt.plot(self.y, result)
            plt.show()
        return result

    def big_l(self, index: str, show: bool = False) -> np.array:
        result = self.R[index] - self.c.J_plus + (self.c.J_plus - self.c.J_minus) * (0.5 - self.y)
        # print(result[0], result[-1])
        # result = result[::-1]
        if show:
            plt.plot(self.y, result)
            plt.show()
        return result

    def big_j(self, index: str, show: bool = False) -> np.array:
        result = np.exp((self.c.E_A_top_underline * (self.Z[index] - 1)) / self.Z[index])
        if show:
            plt.plot(self.y, result)
            plt.show()
        return result

    def tau_0_top_underline(self, index: str, show: bool = False) -> np.array:
        result = 1 / (self.Z[index] * self.big_j(index))
        if show:
            plt.plot(self.y, result)
            plt.show()
        return result

    # page 2 upper (23)
    def delta(self, index: str, show: bool = False) -> np.array:
        result = (self.a_1_1[index] * self.a_2_2[index]) - (self.a_1_2[index] ** 2)
        if show:
            plt.plot(self.y, result)
            plt.show()
        return result

    def big_i_wavy_underline(self, index: str, show: bool = False) -> np.array:
        result = (self.a_1_1[index] + self.a_2_2[index])
        if show:
            plt.plot(self.y, result)
            plt.show()
        return result

    def big_k_big_i_wavy_underline(self, index: str, show: bool = False) -> np.array:
        result = (self.c.W ** -1) + (self.c.k_cap / 3) * self.c.Re * self.big_i_wavy_underline(
            index)
        if show:
            plt.plot(self.y, result)
            plt.show()
        return result

    def big_capital_l(self, index: str, show: bool = False) -> np.array:
        result = ((self.c.A_r_top_wavy_underline * self.Z[index] * self.a_1_2[index]) +
                  (self.c.A_m_top_wavy_underline * self.c.sigma_m * (1 + self.c.m) * self.L[index]))
        if show:
            plt.plot(self.y, result)
            plt.show()
        return result

    def big_b_2_2(self, index: str, show: bool = False) -> np.array:
        result = self.big_k_big_i_wavy_underline(index) / self.tau_0_top_underline(index)
        if show:
            plt.plot(self.y, result)
            plt.show()
        return result

    def small_sigma(self, show: bool = False):
        sigma = np.zeros(self.y.shape)
        sigma[1:-1] = self.sigma['n'][1:-1] + \
                      (self.a_1_2['n+l+1'][1:-1] * self.r * (self.u['n+l+1'][:-2] - self.u['n+l+1'][2:]))

        delimetr = 1 + self.tau * (
                    self.big_k_big_i_wavy_underline('n+l')[1:-1] / self.tau_0_top_underline('n+l')[1:-1] +
                    self.big_k_big_i_wavy_underline('n')[1:-1] / self.tau_0_top_underline('n')[1:-1]) / 2

        sigma[1:-1] /= delimetr
        if show:
            plt.plot(self.y, sigma)
            plt.show()
        return sigma

    def small_a_1_1(self, show: bool = False) -> np.array:
        result = self.small_sigma() + self.a_2_2['n+l+1']
        if show:
            plt.plot(self.y, result)
            plt.show()
        return result

    def small_a_2_2_n_l_1(self, show: bool = False):
        first = self.c.betta * self.c.Re * self.tau * ((self.delta('n+l') / self.tau_0_top_underline('n+l')) +
                                                       (self.delta('n') / self.tau_0_top_underline('n'))) / 2
        second = self.tau * (self.big_k_big_i_wavy_underline('n+l') / self.tau_0_top_underline('n+l') +
                             self.big_k_big_i_wavy_underline('n') / self.tau_0_top_underline('n')) / 2
        result = (self.a_2_2['n'] + first) / (1 + second)

        if show:
            plt.plot(self.y, result)
            plt.show()

        return result

    def update_matrix(self):
        for k in range(250000):
            for j in range(3):
                self.a_2_2['n+l+1'] = self.small_a_2_2_n_l_1()

                capital_big_l = (self.big_capital_l('n+l') + self.big_capital_l('n')) / 2
                b_22 = 1 + (self.tau * (self.big_b_2_2('n+l') + self.big_b_2_2('n')) / 2)
                f_0 = self.u['n'] + self.tau * self.c.A_0 - self.c.sigma_m * (1 + self.c.m) * \
                      (self.c.J_plus - self.c.J_minus)
                self.A[:, 0, 1] = self.Z['(n+l)']
                self.A[:, 0, 2] = self.a_1_2['(n+l)']
                self.A[:, 0, 3] = self.c.sigma_m * (1 + self.c.m)
                self.A[:, 1, 0] = self.a_2_2['n+l+1']
                self.A[:, 2, 0] = capital_big_l
                self.A[:, 2, 2] = - 2 / (self.h * self.c.Pr)
                self.A[:, 3, 0] = (1 + self.c.m)
                self.A[:, 3, 3] = -2 * self.c.b_m / self.h
                self.A = (self.r / 2) * self.A
                # print(np.linalg.det(self.A[, :, :]))
                # det_A = np.array([np.linalg.det(self.A[i, :, :]) for i in range(self.num)])

                self.C = self.A
                self.C[:, 2, 2] = - self.C[:, 2, 2]
                self.C[:, 3, 3] = - self.C[:, 3, 3]
                self.C = - self.C

                self.B[:, 0, 0] = 1
                self.B[:, 0, 1] = self.tau * self.c.theta_top_underline
                self.B[:, 1, 1] = b_22
                self.B[:, 2, 2] = 1 + 2 * self.r / (self.h * self.c.Pr)
                self.B[:, 3, 3] = 1 + 2 * (self.r * self.c.b_m / self.h)

                self.F[:, 0] = f_0
                self.F[:, 1] = self.a_1_2['n']
                self.F[:, 2] = self.Q['n']
                self.F[:, 3] = self.R['n']
                # print(self.A, '\n%%%%%%%%%%%%%%%%%%%%%%%%%%%')

                d_cup_underline = self.Z['(n+l)'][0] - self.c.theta_top_underline * self.h
                self.X[0, 1, 1] = self.Z['(n+l)'][0] / d_cup_underline
                self.X[0, 1, 2] = self.a_1_2['(n+l)'][0] / d_cup_underline
                self.X[0, 1, 3] = self.c.sigma_m * (1 + self.c.m) / d_cup_underline

                self.Y[0, 1] = (self.h * (self.c.big_a((self.n + 1)*self.tau) +
                                self.c.sigma_m * (1 + self.c.m) * (
                                        self.c.J_plus - self.c.J_minus) * self.h)) / d_cup_underline

                d_wavy_underline = self.Z['(n+l)'][-1] + self.c.theta_top_underline * self.h
                self.X_[-1, 1, 1] = - self.Z['(n+l)'][-1] / d_wavy_underline
                self.X_[-1, 1, 2] = - self.a_1_2['(n+l)'][-1] / d_wavy_underline
                self.X_[-1, 1, 3] = - self.c.sigma_m * (1 + self.c.m) / d_wavy_underline

                self.Y_[-1, 1] = (- self.h * (self.c.big_a((self.n + 1)) +
                                  self.c.sigma_m * (1 + self.c.m) * (
                                          self.c.J_plus - self.c.J_minus) * self.h)) / d_wavy_underline

                for i in range(2, self.num):
                    first_ = np.linalg.inv((self.C[-i, :, :] @ (self.X_[-i + 1, :, :]) + self.B[-i, :, :]))
                    self.X_[-i, :, :] = - first_ @ (self.A[-i, :, :])
                    self.Y_[-i, :] = first_ @ (self.F[-i, :] - self.C[-i, :, :] @ (self.Y_[-i+1, :]))

                for i in range(1, self.num):
                    first_ = np.linalg.inv((self.A[i, :, :] @ (self.X[i - 1, :, :]) + self.B[i, :, :]))
                    self.X[i, :, :] = - first_ @ (self.C[i, :, :])
                    self.Y[i, :] = first_ @ (self.F[i, :] - self.C[i, :, :] @ (self.Y[i - 1, :]))

                real_x = np.concatenate((self.X[: self.num // 2, :, :], self.X_[self.num // 2:, :, :]), axis=0)
                real_y = (np.concatenate((self.Y[: self.num // 2, :], self.Y_[self.num // 2:, :]), axis=0))

                u_n_l_1 = np.zeros((self.num, 4)).astype(np.float64)
                u_n_l_1[-1, :] = np.linalg.inv(np.eye(4) - (real_x[-1, :, :].dot(real_x[-2, :, :])))\
                    .dot(real_y[-1, :] + (real_x[-1, :, :].dot(real_y[-2, :])))

                for i in range(2, self.num):
                    u_n_l_1[-i, :] = real_x[- i, :, :].dot(u_n_l_1[-i + 1, :]) + real_y[- i, :]

                # plt.plot(self.y[1:-1], self.X[:, 0, 0][1:-1], label='(0, 0)')
                # plt.plot(self.y[1:-1], self.X[:, 0, 1][1:-1], label='(0, 1)')
                # plt.plot(self.y[1:-1], self.X[:, 0, 2][1:-1], label='(0, 2)')
                # plt.plot(self.y[1:-1], self.X[:, 1, 0][1:-1], label='(1, 0)')
                # plt.plot(self.y[1:-1], self.X[:, 1, 1][1:-1], label='(1, 1)')
                # plt.plot(self.y[1:-1], self.X[:, 1, 2][1:-1], label='(1, 2)')
                # plt.plot(self.y[1:-1], self.X[:, 2, 0][1:-1], label='(2, 0)')
                # plt.plot(self.y[1:-1], self.X[:, 2, 1][1:-1], label='(2, 1)')
                # plt.plot(self.y[1:-1], self.X[:, 2, 2][1:-1], label='(2, 2)')

                # plt.plot(self.y, real_x[:, 0, 0], label='(0, 0)')
                # plt.plot(self.y, real_x[:, 0, 1], label='(0, 1)')
                # plt.plot(self.y, real_x[:, 0, 2], label='(0, 2)')
                # plt.plot(self.y, real_x[:, 0, 3], label='(0, 3)')
                # plt.plot(self.y, real_x[:, 1, 0], label='(1, 0)')
                # plt.plot(self.y, real_x[:, 1, 1], label='(1, 1)')
                # plt.plot(self.y, real_x[:, 1, 2], label='(1, 2)')
                # plt.plot(self.y, real_x[:, 1, 3], label='(1, 3)')
                # plt.plot(self.y, real_x[:, 2, 0], label='(2, 0)')
                # plt.plot(self.y, real_x[:, 2, 1], label='(2, 1)')
                # plt.plot(self.y, real_x[:, 2, 2], label='(2, 2)')
                # plt.plot(self.y, real_x[:, 2, 3], label='(2, 3)')
                # plt.plot(self.y, real_x[:, 3, 0], label='(3, 0)')
                # plt.plot(self.y, real_x[:, 3, 1], label='(3, 1)')
                # plt.plot(self.y, real_x[:, 3, 2], label='(3, 2)')
                # plt.plot(self.y, real_x[:, 3, 3], label='(3, 3)')
                #
                # plt.plot(self.y, real_y[:, 0], label='0')
                # plt.plot(self.y, real_y[:, 1], label='0')
                # plt.plot(self.y, real_y[:, 2], label='0')
                # plt.plot(self.y, real_y[:, 3], label='0')
                #
                # plt.plot(self.y, self.X_[:, 0, 0], label='(0, 0)')
                # plt.plot(self.y, self.X_[:, 0, 1], label='(0, 1)')
                # plt.plot(self.y, self.X_[:, 0, 2], label='(0, 2)')
                # plt.plot(self.y, self.X_[:, 1, 0], label='(1, 0)')
                # plt.plot(self.y, self.X_[:, 1, 1], label='(1, 1)')
                # plt.plot(self.y, self.X_[:, 1, 2], label='(1, 2)')
                # plt.plot(self.y, self.X_[:, 2, 0], label='(2, 0)')
                # plt.plot(self.y, self.X_[:, 2, 1], label='(2, 1)')
                # plt.plot(self.y, self.X_[:, 2, 2], label='(2, 2)')

                # plt.plot(self.y, self.A[:, 0, 1], label='(0, 1)')
                # plt.plot(self.y, self.A[:, 0, 2], label='(0, 2)')
                # plt.plot(self.y, self.A[:, 1, 0], label='(1, 0)')
                # plt.plot(self.y, self.A[:, 2, 0], label='(2, 0)')
                # plt.legend()
                # plt.show()
                #
                # plt.plot(self.y, self.L['n+l+1'])
                # plt.plot(self.y, det_A)
                # plt.plot(self.y, det_B)
                # plt.plot(self.y, b_22)
                # plt.plot()
                # plt.show()

                self.Q['n+l+1'] = u_n_l_1[:, 2]
                self.R['n+l+1'] = u_n_l_1[:, 3]
                self.Z['n+l+1'] = self.big_z('n+l+1')
                self.L['n+l+1'] = self.big_l('n+l+1')
                self.a_1_2['n+l+1'] = u_n_l_1[:, 1]
                self.a_2_2['n+l+1'] = self.a_2_2['n+l+1']
                self.u['n+l+1'] = u_n_l_1[:, 0]
                self.a_1_1['n+l+1'] = self.small_a_1_1()

                self.Q['n+l'] = self.Q['n+l+1']
                self.R['n+l'] = self.R['n+l+1']
                self.Z['n+l'] = self.Z['n+l+1']
                self.L['n+l'] = self.L['n+l+1']
                self.a_1_2['n+l'] = self.a_1_2['n+l+1']
                self.a_2_2['n+l'] = self.a_2_2['n+l+1']
                self.u['n+l'] = self.u['n+l+1']
                self.a_1_1['n+l'] = self.a_1_1['n+l+1']

                self.Q['(n+l)'] = (self.Q['n+l'] + self.Q['n']) / 2
                self.R['(n+l)'] = (self.R['n+l'] + self.R['n']) / 2
                self.Z['(n+l)'] = (self.Z['n+l'] + self.Z['n']) / 2
                self.L['(n+l)'] = (self.L['n+l'] + self.L['n']) / 2
                self.a_1_2['(n+l)'] = (self.a_1_2['n+l'] + self.a_1_2['n']) / 2
                self.a_2_2['(n+l)'] = (self.a_2_2['n+l'] + self.a_2_2['n']) / 2
                self.a_1_1['(n+l)'] = (self.a_1_1['n+l'] + self.a_1_1['n']) / 2
                self.u['(n+l)'] = (self.u['n+l'] + self.u['n']) / 2
                # plt.plot(self.y, self.u['n+l+1'])
                # plt.show()

            self.n += 1
            self.u['n'] = self.u['n+l+1']
            self.Q['n'] = self.Q['n+l+1']
            self.R['n'] = self.R['n+l+1']
            self.Z['n'] = self.Z['n+l+1']
            self.L['n'] = self.L['n+l+1']
            self.a_1_2['n'] = self.a_1_2['n+l+1']
            self.a_2_2['n'] = self.a_2_2['n+l+1']
            self.a_1_1['n'] = self.a_1_1['n+l+1']
            self.sigma['n'] = self.small_sigma()

            self.Q['n+l'] = self.Q['n']
            self.R['n+l'] = self.R['n']
            self.Z['n+l'] = self.big_z('n')
            self.L['n+l'] = self.big_l('n')

            self.Q['(n+l)'] = (self.Q['n+l'] + self.Q['n']) / 2
            self.R['(n+l)'] = (self.R['n+l'] + self.R['n']) / 2
            self.Z['(n+l)'] = (self.Z['n+l'] + self.Z['n']) / 2
            self.L['(n+l)'] = (self.L['n+l'] + self.L['n']) / 2
            self.a_1_2['(n+l)'] = (self.a_1_2['n+l'] + self.a_1_2['n']) / 2
            self.a_2_2['(n+l)'] = (self.a_2_2['n+l'] + self.a_2_2['n']) / 2
            self.a_1_1['(n+l)'] = (self.a_1_1['n+l'] + self.a_1_1['n']) / 2
            self.u['(n+l)'] = (self.u['n+l'] + self.u['n']) / 2
            # plt.plot(self.y, self.u['n+l+1'])
            # plt.show()
        return self.u['n+l+1']


def main(grid_num: int = 150, time_mult: float = 1/100):
    a = Iteration(num=grid_num, time_mult=time_mult)
    # plt.plot(a.y, a.update_matrix())
    a.update_matrix()


if __name__ == '__main__':
    main(grid_num=100, time_mult=1)

