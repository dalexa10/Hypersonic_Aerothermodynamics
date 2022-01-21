import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib import rcParams

plt.rc('font', family='serif')
plt.rc('font', size=11)
plt.rc('axes', labelsize=11)


def beta_Ma_fn(beta, Ma, gamma):
    """
    Computes the theta, beta, Mach function
    :param beta: oblique shock angle
    :param Ma: Mach number
    :param gamma: adiabatic coefficient
    """
    return (2/np.tan(beta)) * (Ma**2 * (np.sin(beta))**2 - 1) / \
           ((Ma**2 * (gamma + np.cos(2 * beta))) + 2)

def theta_beta_Ma_chart_classic(Ma, gamma):
    """
    Computes and generates the data of the theta-beta-Ma function to be plotted
    :param Ma: Mach number
    :param gamma: adiabatic coefficient
    :return: data: (dict) with beta and thetha values for different Mach numbers
    """
    data = dict()
    for i in range(len(Ma)):
        start_val = fsolve(lambda beta: beta_Ma_fn(beta, Ma[i], gamma), 0.1)
        beta_angles = np.linspace(start_val, np.pi/2, 100, endpoint=True)
        theta_angles = np.arctan(beta_Ma_fn(beta_angles, Ma[i], gamma))
        data[str(Ma[i])] = {
                        "beta": beta_angles * 180/np.pi,
                        "theta": theta_angles * 180/np.pi}

    return data

def theta_beta_Ma_chart_classic_gamma(Ma, gamma):
    """
    Computes and generates the data of the theta-beta-Ma function to be plotted
    :param Ma: Mach number
    :param gamma: adiabatic coefficient
    :return: data: (dict) with beta and thetha values for different Mach numbers
    """
    data = dict()
    for i in range(len(gamma)):
        start_val = fsolve(lambda beta: beta_Ma_fn(beta, Ma, gamma[i]), 0.1)
        beta_angles = np.linspace(start_val, np.pi/2, 100, endpoint=True)
        theta_angles = np.arctan(beta_Ma_fn(beta_angles, Ma, gamma[i]))
        data[str(gamma[i])] = {
                        "beta" : beta_angles * 180/np.pi,
                        "theta": theta_angles * 180/np.pi}

    return data

def theta_beta_Hyper(theta, gamma):
    """
    Computes the beta values for given theta in the hypersonic limit
    :param theta: (np.array), deflection angle (wedge angle)
    :param gamma: (float), adiabatic coefficient
    :return:
    """
    return theta * (gamma + 1) /2


# First plot
gamma1 = 1.4
Ma1 = np.array([2, 5, 10, 25])
theta_hyper1 = np.linspace(0, 50, 100)
data_classic1 = theta_beta_Ma_chart_classic(Ma1, gamma1)
beta_hyper1 = theta_beta_Hyper(theta_hyper1, gamma1)

fig, ax = plt.subplots()
for k in data_classic1.keys():
    ax.plot(data_classic1[k]['theta'], data_classic1[k]['beta'], label='Mach = {}'.format(k))
ax.plot(theta_hyper1, beta_hyper1, '-.', label='Hypersonic Limit')

ax.set_xlim([0, 50])
ax.set_ylim([0, 90])
ax.set_yticks(np.arange(0, 100, 10))
ax.set_xlabel('Theta [deg]')
ax.set_ylabel('Beta [deg]')
ax.set_title('Theta - Beta -Mach chart')
ax.legend(loc='best')
plt.savefig('Figures/theta_beta_classic.svg')
plt.show()

# Second plot
gamma2 = np.array([1.1, 1.2, 1.3, 1.4])
Ma2 = 25
data_gamma2 = theta_beta_Ma_chart_classic_gamma(Ma2, gamma2)

fig, ax = plt.subplots()
for k in data_gamma2.keys():
    ax.plot(data_gamma2[k]['theta'], data_gamma2[k]['beta'], label=r'$\gamma$ = {}'.format(k))

ax.set_xlim([0, 70])
ax.set_ylim([0, 90])
ax.set_yticks(np.arange(0, 100, 10))
ax.set_xlabel('Theta [deg]')
ax.set_ylabel('Beta [deg]')
ax.set_title('Theta - Beta -Mach chart (for Mach = 25)')
ax.legend(loc='best')
plt.savefig('Figures/theta_beta_gamma.svg')
plt.show()
