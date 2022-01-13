import math

import matplotlib.pyplot as plt
import numpy as np


class Stress():
    def __init__(self, A_B=5):
        self.A_B = A_B  # a/b

    def sigma_x_p(self, x_a, y_a):
        y_b = y_a * self.A_B  # (y/a) * (a/b) = y/b
        return ((-3 / 4) * (y_b) * self.A_B ** 2) * (1 - x_a ** 2 + (2 / 3) * y_a ** 2 - (2 / 5) * (1 / self.A_B) ** 2)

    def sigma_y_p(self, x_a, y_a):
        y_b = y_a * self.A_B
        return (1 / 2) + (3 / 4) * y_b - (1 / 4) * (y_b ** 3)

    def tau_xy_p(self, x_a, y_a):
        x_b = x_a * self.A_B
        y_b = y_a * self.A_B
        return (3 / 4) * x_b * (1 - y_b ** 2)

    def tau_max_p(self, x_a, y_a):
        sigma_x = self.sigma_x_p(x_a, y_a)
        sigma_y = self.sigma_y_p(x_a, y_a)
        tau_x_y = self.tau_xy_p(x_a, y_a)
        res = math.sqrt(((sigma_x - sigma_y) / 2) ** 2 + tau_x_y ** 2)
        return res

    def sigma_1_p(self, x_a, y_a):
        sigma_x = self.sigma_x_p(x_a, y_a)
        sigma_y = self.sigma_y_p(x_a, y_a)
        tau_max = self.tau_max_p(x_a, y_a)

        return (sigma_x + sigma_y) / 2 + tau_max

    def sigma_2_p(self, x_a, y_a):
        sigma_x = self.sigma_x_p(x_a, y_a)
        sigma_y = self.sigma_y_p(x_a, y_a)
        tau_max = self.tau_max_p(x_a, y_a)

        return (sigma_x + sigma_y) / 2 - tau_max

    def means_1D(self):
        # generate 1D vector of both x & y
        #a = 200  # for example
        #b = a / self.A_B
        x_mean = np.divide(np.linspace(-1, 1), 1)
        y_mean = np.divide(np.linspace(-1 / self.A_B, 1 / self.A_B), 1)
        return x_mean, y_mean

    def means_2D(self):
        x_mean, y_mean = self.means_1D()
        return np.meshgrid(x_mean, y_mean) # X, Y

    def vectorized_tau(self):
        vector1 = np.vectorize(self.tau_max_p)
        return vector1(self.means_2D()[0],self.means_2D()[1])

    def vectorized_sigma_1_2(self):
        vector1 = np.vectorize(self.sigma_1_p)
        vector2 = np.vectorize(self.sigma_2_p)
        return vector1(self.means_2D()[0],self.means_2D()[1]), vector2(self.means_2D()[0],self.means_2D()[1])

    def plot_sigmas(self):
        tau_max = self.vectorized_tau()
        sigma_1, sigma_2 = self.vectorized_sigma_1_2()
        X,Y = self.means_2D()

        fig = plt.figure()
        fig.suptitle('rel . Haupt-/Schub-Spannungen')
        ax1 = plt.subplot(3, 1, 1)
        fig.add_subplot(ax1)
        ax1.set_ylabel('y/a')
        ax1.set_xlabel('x/a')
        ax1.set_title(r'$\sigma_1 / p$')
        ctr1 = ax1.contour(X,Y,sigma_1, cmap='jet')
        plt.tight_layout()
        plt.colorbar(ctr1)

        ax2 = plt.subplot(3, 1, 2)
        fig.add_subplot(ax2)
        ax2.set_ylabel('y/a')
        ax2.set_xlabel('x/a')
        ax2.set_title(r'$\sigma_2 / p$')
        rev_map = plt.cm.get_cmap('jet').reversed()
        ctr2 = ax2.contour(X,Y,sigma_2, cmap=rev_map)
        plt.tight_layout()
        plt.colorbar(ctr2, ax = ax2, shrink = 0.75)

        ax3 = plt.subplot(3, 1, 3)
        fig.add_subplot(ax3)
        ax3.set_ylabel('y/a')
        ax3.set_xlabel('x/a')
        ax3.set_title(r'real. max. pos. Schubspannung $\tau_{max} / p$')
        ctr3 = ax3.contour(X,Y,tau_max, cmap='jet')
        plt.colorbar(ctr3)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    x = Stress()
    x.plot_sigmas()
