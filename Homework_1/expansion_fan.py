import numpy as np
import math
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def Prandl_Meyer(Ma, gamma):
    """
    Return value of Prandl-Meyer function
    :param Ma: (float), Mach Number
    :param gamma: (float), adiabatic constant
    :return:
    """
    return np.sqrt((gamma+1)/(gamma-1)) * np.arctan(np.sqrt(((gamma - 1) * (Ma**2 -1))/(gamma + 1)))\
        - np.arctan(np.sqrt(Ma**2 - 1))

def Prandl_Meyer_downstream(v_Ma1, theta, deg=True):
    if deg is True:
        theta = math.radians(theta)
    v_Ma2 = theta + v_Ma1
    return v_Ma2

def Prandl_Meyer_inv(Ma1, gamma, theta):
    v_Ma1 = Prandl_Meyer(Ma1, gamma)
    v_Ma2 = Prandl_Meyer_downstream(v_Ma1, theta)
    Ma2 = fsolve(lambda Ma: v_Ma2 - Prandl_Meyer(Ma, gamma), Ma1 + 0.2)
    return Ma2[0]

def isentropic_relations(Ma1, Ma2, gamma):
    T2_T1 = ((1 + (gamma-1)/2) * Ma1**2)/((1 + (gamma-1)/2) * Ma2**2)
    P2_P1 = (T2_T1)**(gamma/(gamma-1))
    return T2_T1, P2_P1

def Prandl_Meyer_Hyper(Ma, gamma):
    v_Ma = (np.sqrt((gamma + 1)/(gamma - 1)) - 1) * np.pi/2 - 2/((gamma - 1) * Ma)
    return v_Ma

def Mach_down_hyper(Ma1, gamma, theta, deg=True):
    if deg is True:
        theta = math.radians(theta)
    inv_Ma2 = (1/Ma1) - theta * (gamma - 1)/2
    Ma2 = 1/inv_Ma2
    return Ma2

def isentropic_relations_hyper(Ma1, Ma2, gamma):
    P2_P1 = (Ma1/Ma2)**(2 * gamma / (gamma - 1))
    T2_T1 = (P2_P1)**((gamma - 1) / gamma)
    return T2_T1, P2_P1


theta = 10
gamma = 1.4
Ma = np.array([3, 5, 10, 25])

# Results - Classical Theory
data_classic = dict()
data_hyper = dict()

for i in range(len(Ma)):
    v_M1 = Prandl_Meyer(Ma[i], gamma)
    v_M2 = Prandl_Meyer_downstream(v_M1, theta)
    M2 = Prandl_Meyer_inv(Ma[i], gamma, theta)
    T2_T1, P2_P1 = isentropic_relations(Ma[i], M2, gamma)
    data_classic[str(Ma[i])] = {"v_Ma1": v_M1,
                                "v_Ma2": v_M2,
                                "Ma_2": M2,
                                "T2_T1": T2_T1,
                                "P2_P1": P2_P1}

for i in range(len(Ma)):
    v_M1_h = Prandl_Meyer_Hyper(Ma[i], gamma)
    v_M2_h = Prandl_Meyer_downstream(v_M1_h, theta)
    M2_h = Mach_down_hyper(Ma[i], gamma, theta)
    T2_T1_h, P2_P1_h = isentropic_relations_hyper(Ma[i], M2_h, gamma)
    data_hyper[str(Ma[i])] = {"v_Ma1": v_M1_h,
                              "v_Ma2": v_M2_h,
                              "Ma_2": M2_h,
                              "T2_T1": T2_T1_h,
                              "P2_P1": P2_P1_h}

#%%

# Table1 = pd.DataFrame(data_classic)
# Table2 = pd.DataFrame(data_hyper)
#
# # Plot
# fig, ax = plt.subplots(1, 3)
#
# ax[0].set_box_aspect(1)
# ax[0].plot(Ma, Table1.iloc[0, :], 'b', label='Upstream P-M (Classic)')
# ax[0].plot(Ma, Table1.iloc[1, :], 'g', label='Downstream P-M (Classic)')
# ax[0].plot(Ma, Table2.iloc[0, :], 'r', label='Upstream P-M (Hypersonics)')
# ax[0].plot(Ma, Table2.iloc[1, :], 'k', label='Downstream P-M (Hypersonics)')
# ax[0].set_ylabel(r'$\nu(Ma)$')
# ax[0].set_xlabel('Ma')
# ax[0].set_title('Prandl-Meyer vs Ma')
# ax[0].legend(loc='best')
#
# ax[1].set_box_aspect(1)
# ax[1].plot(Ma, Table1.iloc[2, :], 'b', label='Dowstream Mach (Classic)')
# ax[1].plot(Ma, Table2.iloc[2, :], 'r', label='Downstream Mach (Hypersonics)')
# ax[1].set_ylabel(r'$Ma_{2}$')
# ax[1].set_xlabel(r'$Ma_{1}$')
# ax[1].set_title('Downstream vs upstream Mach')
# ax[1].legend(loc='best')
#
# ax[2].set_box_aspect(1)
# ax[2].plot(Ma, Table1.iloc[3, :], 'b', label='Temperature ratio (Classic)')
# ax[2].plot(Ma, Table1.iloc[4, :], 'g', label='Pressure ratio (Classic)')
# ax[2].plot(Ma, Table2.iloc[3, :], 'r', label='Temperature ratio (Hypersonics)')
# ax[2].plot(Ma, Table2.iloc[4, :], 'k', label='Pressure ratio (Hypersonics)')
# ax[2].set_ylabel('P2/P1 or T2/T1')
# ax[2].set_xlabel('Ma')
# ax[2].set_title('Isentropic Relations')
# ax[2].legend(loc='best')
#
# # Just a test if git works
#
# plt.savefig('Figures/prandl_meyer_outcomes.svg')
# plt.show()

#%%
# Just in case: Ma2
Ma = 6
gamma = 1.4
theta = 20
Ma2 = Prandl_Meyer_inv(Ma1, gamma, theta)