import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from ambiance import Atmosphere
import math

# Oblique Shock Computation
def Mach1_normal(Ma, beta):
    Ma1n = Ma * np.sin(beta)
    return Ma1n

def Mach2(Ma2n, beta, theta):
    Ma2 = Ma2n/np.sin(beta - theta)
    return Ma2

def theta_beta_Ma_fn(theta, beta, Ma, gamma=1.4):
    return np.tan(theta) - (2/np.tan(beta)) * (Ma**2 * (np.sin(beta))**2 - 1) / \
           ((Ma**2 * (gamma + np.cos(2 * beta))) + 2)

def beta_calculation(Ma, theta):
    """
    Computes the beta angle [rad]
    :param Ma: (float) upstream mach number
    :param theta: (float), theta angle [rad]
    :return: beta (float) [rad]
    """
    beta = fsolve(lambda beta: theta_beta_Ma_fn(theta, beta, Ma), 0.1)
    beta = beta[0]
    return beta


class Post_Shock:
    """
    Computes the Post Shock Relations (Warning! Ma1 is the NORMAL pre-shock Mach number)
    """
    def __init__(self, Ma1, gamma):
        self.Ma1 = Ma1
        self.gamma = gamma
        self.rho21 = self.rho_ratio()
        self.p21 = self.pressure_ratio()
        self.T21 = self.temperature_ratio()
        self.U21 = 1 / self.rho21
        self.Ma2 = self.Mach_downstream()
        self.Po21 = self.total_pressure_ratio()


    def rho_ratio(self):
        rho21 = ((self.gamma + 1) * self.Ma1**2) / (2 + (self.gamma - 1) * self.Ma1**2)
        return rho21

    def pressure_ratio(self):
        p21 = 1 + (2 * self.gamma / (self.gamma + 1)) * (self.Ma1**2 - 1)
        return p21

    def temperature_ratio(self):
        T21 = self.p21/self.rho21
        return T21

    def Mach_downstream(self):
        Ma2 = np.sqrt((2 + (self.gamma - 1) * self.Ma1**2) / (2 * self.gamma * self.Ma1**2 + 1 - self.gamma))
        return Ma2

    def total_pressure_ratio(self):
        Po21 = ((((self.gamma + 1) * self.Ma1**2) / ((self.gamma - 1) * self.Ma1**2 + 2))**(self.gamma / (self.gamma - 1))) * \
               (((self.gamma + 1)/(2 * self.gamma * self.Ma1**2 - (self.gamma-1)))**(1 / (self.gamma - 1)))
        return Po21


