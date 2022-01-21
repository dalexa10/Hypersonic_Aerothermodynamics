import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from ambiance import Atmosphere
import math

class Post_Fan():
    def __init__(self, Ma1, theta, gamma):
        self.Ma1 = Ma1
        self.theta = theta
        self.gamma = gamma
        self.Ma2 = self.Prandl_Meyer_inv()
        self.T21 = self.temperature_ratio()
        self.p21 = self.pressure_ratio()


    def Prandl_Meyer(self, Ma):
        return np.sqrt((self.gamma + 1) / (self.gamma - 1)) * \
               np.arctan(np.sqrt(((self.gamma - 1) * (Ma ** 2 - 1)) / (self.gamma + 1))) - \
               np.arctan(np.sqrt(Ma ** 2 - 1))

    def Prandl_Meyer_downstream(self):
        v_Ma1 = self.Prandl_Meyer(self.Ma1)
        v_Ma2 = self.theta + v_Ma1
        return v_Ma2

    def Prandl_Meyer_inv(self):
        v_Ma2 = self.Prandl_Meyer_downstream()
        Ma2 = fsolve(lambda Ma: v_Ma2 - self.Prandl_Meyer(Ma), self.Ma1 + 0.2)
        return Ma2[0]

    def temperature_ratio(self):
        T21 = (1 + ((self.gamma - 1) / 2) * self.Ma1 ** 2) / (1 + ((self.gamma - 1) / 2) * self.Ma2 ** 2)
        return T21

    def pressure_ratio(self):
        p21 = self.T21**(self.gamma / (self.gamma - 1))
        return p21






