from math import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from scipy.integrate import ode

# --- Font Parameters ---
font = {'size': 22}
rc('font', **font)
rcParams['font.sans-serif'] = 'Arial'
rcParams['font.family'] = 'sans-serif'


def solve_compressible_BL(sol_params, params):
    """
    Solve compressible boundary similarity equations for dP_e/dx = 0
    (Cf'')' + f f'' = 0
    (Cg')' + Pr f g' + Pr C (gamma-1) M^2 f''^2 = 0
    where f'=u/ue, g=T/Te and C=(rho*mu)/(rhoe*mue)=g^(n-1), using the secant method.
    Returns F = [f, f', f'', g, g']
    """
    [Ma_e, Pr, gamma, sigma] = params
    [flag_ad, g_w, eta_max, dt, n_iter, tolerance, q0] = sol_params
    q0_nm1 = q0
    q0_nm2 = q0 + .01
    for i in range(n_iter):
        # solve initial guesses
        eta, F = solve_ode(q0_nm1, flag_ad, g_w, params, eta_max, dt)
        fp_nm1 = F[-1, 1]
        g_nm1 = F[-1, 3]
        eta, F = solve_ode(q0_nm2, flag_ad, g_w, params, eta_max, dt)
        fp_nm2 = F[-1, 1]
        g_nm2 = F[-1, 3]
        # secant method
        fpp_n = (q0_nm2[0] * (fp_nm1 - 1) - q0_nm1[0]*(fp_nm2 - 1)) / ((fp_nm1 - 1) - (fp_nm2 - 1))
        gp_n = (q0_nm2[1] * (g_nm1 - 1) - q0_nm1[1]*(g_nm2 - 1)) / ((g_nm1 - 1) - (g_nm2 - 1))
        # update initial guess
        q0_nm2 = q0_nm1
        q0_nm1 = [fpp_n, gp_n]
        # diff magnitude and far-field error
        #diff = sqrt( (q0_nm2[0]-q0_nm1[0])**2 - (q0_nm2[1]-q0_nm1[1])**2 )
        error = sqrt((fp_nm1 - 1)**2 + (g_nm1 - 1)**2)
        print('  ')
        print(' Iteration ' + str(i) + ' : error = ' + str(error))
        print(' fpp(0) = ' + str(q0_nm1[0]) + ' , gp(0) = ' + str(q0_nm1[1]))
        print(' fp(inf) = ' + str(fp_nm1) + ' , g(inf) = ' + str(g_nm1))
        # break if not converged
        if np.isnan(q0_nm1[0]) == True or np.isnan(q0_nm1[1]) == True: raise Exception('Failed to converge! Try changing initial guess...')
        if np.isinf(q0_nm1[0]) == True or np.isinf(q0_nm1[1]) == True: raise Exception('Failed to converge! Try changing initial guess...')
        # break if tolerance is met
        if error < tolerance: break
    # return
    return eta, F


def solve_ode(q0, flag_ad, g_w, params, eta_max, dt):
    """ Solve ODE """
    if flag_ad:
        ic = np.array([0, 0, q0[0], q0[1], 0])  # init. cond. if adiabatic
    else:
        ic = np.array([0, 0, q0[0], g_w, q0[1]])  # init. cond. if isothermal
    eta_0 = 0
    # solve
    solver = ode(RHS).set_integrator('dopri5')
    solver.set_initial_value(ic, eta_0).set_f_params(params)
    eta = [eta_0]
    F = ic[None, :]
    while solver.successful() and solver.t < eta_max:
        eta.append(solver.t + dt)
        F = np.concatenate((F, solver.integrate(solver.t + dt)[None, :]))
    # return
    return eta, F


def RHS(t, F, params):
    """ Compute RHS of compressible boundary layer similarity solution"""
    Fp = np.zeros((5, 1))
    [Ma_e, Pr, gamma, sigma] = params
    # compute RHS
    Fp[0] = F[1]  # f'
    Fp[1] = F[2]  # f''
    Fp[2] = -1/F[3]**(sigma-1)*(F[0]*F[2] + (sigma-1)*F[3]**(sigma-2)*F[4]*F[2])  # f'''
    Fp[3] = F[4]  # g'
    Fp[4] = -1/F[3]**(sigma-1)*((sigma-1)*F[3]**(sigma-2)*F[4]**2 + Pr*F[0]*F[4]
                                + Pr*F[3]**(sigma-1)*(gamma-1)*Ma_e**2*F[2]**2)  # g''

    return Fp  # [f', f'', f''', g', g'']

#%%
# -----------------------------
# ----- ISOTHERMAL CASE -------
# -----------------------------

def main():

    # ----- Input Parameters ------
    Ma_e = 5  # edge Mach number
    Pr = 0.72  # Prandtl number
    sigma = 0.75  # power law coefficient for viscosity mu/mu0=(T/T0)^n
    gamma = 7 / 5  # adiabatic coefficient
    flag_ad = 0  # Flag for wall thermal boundary condition: 1=adiabatic wall; 0=isothermal wall;
    g_w = 6  # dimensionless wall temperature (Tw/Te)
    eta_max = 15  # maximum eta coordinate for integration
    dt = .01  # ode t
    n_iter = 20  # max number of iterations
    tolerance = 1e-5  # tolerance
    q0 = np.array([0.69, 8.2])  # initial guess for f''(0) and g'(0), in video ([0.65, 8]), original 0.60, 1.7]
    #########################################################################
    params = [Ma_e, Pr, gamma, sigma]
    sol_params = [flag_ad, g_w, eta_max, dt, n_iter, tolerance, q0]

    # ---- Solver Calling ----
    eta, F = solve_compressible_BL(sol_params, params)
    m = np.zeros_like(eta)
    m[:] = (F[:, 3] + ((gamma - 1)/2*Ma_e**2*(F[:, 1])**2))/(F[-1, 3] + ((gamma - 1)/2*Ma_e**2*(F[-1, 1])**2))

    # Non dimensional parameters plotting section

    plt.figure(1, figsize=(10, 6))
    plt.plot(F[:, 1], eta, linewidth=1, label=(' $f\'$, velocity'), color='r')
    plt.plot(F[:, 3], eta, linewidth=1, label=('$g$, static temperature '), color='b')
    plt.plot(m, eta, linewidth=1, label=(' $m$, stagnation temperature '), color='g')

    ax = plt.figure(1)
    plt.xlabel('Flow parameters')
    plt.ylabel(r'$\eta$')
    plt.xlim(0, 6)
    plt.ylim(0, 5)
    plt.tick_params(direction='in', length=8)
    plt.legend(loc='upper right', fontsize=20)
    plt.grid(linestyle='--')
    if flag_ad == 1:
        strTemp = str('Adiabatic')
    else:
        strTemp = str(r'$T_w/T_e=%1.1f$' % (g_w, ))
    textstr = (r'$Ma_e=%2d$' % (Ma_e, ) + ', ' +
               r'$\gamma=%1.1f$' % (gamma, ) + ', ' +
               r'$Pr=%1.2f$' % (Pr, ) + ', ' +
               r'$\sigma=%.2f$' % (sigma, ) + ', ' +
               strTemp)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.9, 0.95, textstr, fontsize=14, verticalalignment='top', horizontalalignment='right', bbox=props)
    plt.savefig('./Figures/fig1.svg')
    plt.show()


if __name__ == "__main__":
    main()


#%%
# ------------------------------
# ------ ADIABATIC CASE --------
# ------------------------------

def main():

    # ----- Input Parameters ------
    Ma_e = 5  # edge Mach number
    Pr = 0.72  # Prandtl number
    sigma = 0.75  # power law coefficient for viscosity mu/mu0=(T/T0)^n
    gamma = 7 / 5  # adiabatic coefficient
    flag_ad = 1  # Flag for wall thermal boundary condition: 1=adiabatic wall; 0=isothermal wall;
    g_w = 6  # dimensionless wall temperature (Tw/Te)
    eta_max = 15  # maximum eta coordinate for integration
    dt = .01  # ode t
    n_iter = 20  # max number of iterations
    tolerance = 1e-5  # tolerance
    q0 = np.array([0.69, 8.2])  # initial guess for f''(0) and g'(0), in video ([0.65, 8]), original 0.60, 1.7]
    #########################################################################
    params = [Ma_e, Pr, gamma, sigma]
    sol_params = [flag_ad, g_w, eta_max, dt, n_iter, tolerance, q0]

    # ---- Solver Calling ----
    eta, F = solve_compressible_BL(sol_params, params)
    m = np.zeros_like(eta)
    m[:] = (F[:, 3] + ((gamma - 1)/2*Ma_e**2*(F[:, 1])**2))/(F[-1, 3] + ((gamma - 1)/2*Ma_e**2*(F[-1, 1])**2))

    # Non dimensional parameters plotting section

    plt.figure(2, figsize=(10, 6))
    plt.plot(F[:, 1], eta, linewidth=1, label=(' $f\'$, velocity'), color='r')
    plt.plot(F[:, 3], eta, linewidth=1, label=('$g$, static temperature '), color='b')
    plt.plot(m, eta, linewidth=1, label=(' $m$, stagnation temperature '), color='g')

    ax = plt.figure(2)
    plt.xlabel('Flow parameters')
    plt.ylabel(r'$\eta$')
    plt.xlim(0, 8)
    plt.ylim(0, 5)
    plt.tick_params(direction='in', length=8)
    plt.legend(loc='upper right', fontsize=20)
    plt.grid(linestyle='--')
    if flag_ad == 1:
        strTemp = str('Adiabatic')
    else:
        strTemp = str(r'$T_w/T_e=%1.1f$' % (g_w, ))
    textstr = (r'$Ma_e=%2d$' % (Ma_e, ) + ', ' +
               r'$\gamma=%1.1f$' % (gamma, ) + ', ' +
               r'$Pr=%1.2f$' % (Pr, ) + ', ' +
               r'$\sigma=%.2f$' % (sigma, ) + ', ' +
               strTemp)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.9, 0.95, textstr, fontsize=14, verticalalignment='top', horizontalalignment='right', bbox=props)
    plt.savefig('./Figures/fig2.svg')
    plt.show()


if __name__ == "__main__":
    main()


#%%
# --------------------------------------
# ---------- DIMENSIONAL BL ------------
# --------------------------------------

# ----- Input Parameters ------
Ma_e = 5  # edge Mach number
Pr = 0.72  # Prandtl number
flag_ad = 1
sigma = 0.75  # power law coefficient for viscosity mu/mu0=(T/T0)^n
gamma = 7 / 5  # adiabatic coefficient
eta_max = 15  # maximum eta coordinate for integration
dt = .01  # ode t
n_iter = 20  # max number of iterations
tolerance = 1e-5  # tolerance

# ---- Variable parameters ----
# initial guess for f''(0) and g'(0)
q0 = np.array([0.69, 8.2])  # Adiabatic wall guees
q1 = np.array([0.65, 7.0])  # Isothermal wall Tw/Te = 4
q2 = np.array([0.65, 7.0])  # Isothermal wall Tw/Te = 8

# ---- Solution parameters ----
params = [Ma_e, Pr, gamma, sigma]

sol_params0 = [1, 0, eta_max, dt, n_iter, tolerance, q0]  # Adiabatic wall
sol_params1 = [0, 4, eta_max, dt, n_iter, tolerance, q1]  # Isothermal wall Tw/Te = 4
sol_params2 = [0, 8, eta_max, dt, n_iter, tolerance, q2]  # Isothermal wall Tw/Te = 8

# ---- Solver Calling ----
eta0, F0 = solve_compressible_BL(sol_params0, params)
eta1, F1 = solve_compressible_BL(sol_params1, params)
eta2, F2 = solve_compressible_BL(sol_params2, params)

m0 = np.zeros_like(eta0)
m1 = np.zeros_like(eta1)
m2 = np.zeros_like(eta2)

T_e = 223
p_e = 1000
mu_e = 1.2e-5
Re = 10000
R = 287
rho_e = p_e / (R * T_e)
V_e = Ma_e * np.sqrt(gamma * R * T_e)
x = mu_e * Re / (rho_e * V_e)
cp = 1e3

# Calculate y using midpoint rectangles
y0 = np.zeros(len(eta0))
y1 = np.zeros(len(eta1))
y2 = np.zeros(len(eta2))

factor = x * sqrt(2 / Re)

integral0 = 0
integral1 = 0
integral2 = 0

for j in range(1, len(eta0)):
    integral0 += (eta0[j] - eta0[j-1]) * (F0[j, 3] + F0[j-1, 3]) / 2
    y0[j] = factor * integral0
u0 = V_e * F0[:, 1]
T_0 = T_e * F0[:, 3]
Tw_0 = T_0[0]
T_aw = T_0[0]
rho_w_0 = p_e / (R * Tw_0)
mu_w_0 = mu_e * (Tw_0 / T_e)**sigma
T0_0 = T_0 * (1 + (gamma - 1) / 2 * Ma_e ** 2)
Cw_0 = F0[0, 3]**(sigma - 1)


for j in range(1, len(eta1)):
    integral1 += (eta1[j] - eta1[j-1]) * (F1[j, 3] + F1[j-1, 3]) / 2
    y1[j] = factor * integral1
u1 = V_e * F1[:, 1]
T_1 = T_e * F1[:, 3]
Tw_1 = T_1[0]
rho_w_1 = p_e / (R * Tw_1)
mu_w_1 = mu_e * (Tw_1 / T_e)**sigma
T0_1 = T_1 * (1 + (gamma - 1) / 2 * Ma_e ** 2)
Cw_1 = F1[0, 3]**(sigma - 1)

for j in range(1, len(eta2)):
    integral2 += (eta2[j] - eta2[j-1]) * (F2[j, 3] + F2[j-1, 3]) / 2
    y2[j] = factor * integral2
u2 = V_e * F2[:, 1]
T_2 = T_e * F2[:, 3]
Tw_2 = T_2[0]
rho_w_2 = p_e / (R * Tw_2)
mu_w_2 = mu_e * (Tw_2 / T_e)**sigma
T0_2 = T_2 * (1 + (gamma - 1) / 2 * Ma_e ** 2)
Cw_2 = F2[0, 3]**(sigma - 1)

deltaB0 = x * Re**(-1/2)

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 6))

ax0.plot(F0[:, 1], y0 / deltaB0, linewidth=1, color='b', label='Adiabatic')
ax0.plot(F1[:, 1], y1 / deltaB0, linewidth=1, color='g', label='$T_{w}/T_e$ = 4')
ax0.plot(F2[:, 1], y2 / deltaB0, linewidth=1, color='r', label='$T_{w}/T_e$ = 8')
ax0.set_xlabel('$u/V_e$')
ax0.set_ylabel(r'$y/\delta_{BO}$')
ax0.set_xlim(0, 1.2)
ax0.set_ylim(0, 20)
ax0.grid(linestyle='--')
ax0.tick_params(direction='in', length=8)


ax1.plot(F0[:, 3], y0 / deltaB0, linewidth=1, color='b', label='Adiabatic')
ax1.plot(F0[:, 3] + ((gamma - 1) / 2 * Ma_e ** 2 * (F0[:, 1]) ** 2), y0 / deltaB0, linewidth=1, color='b',
         linestyle='dashed')

ax1.plot(F1[:, 3], y1 / deltaB0, linewidth=1, color='g', label='$T_{w}/T_e$ = 4')
ax1.plot(F1[:, 3] + ((gamma - 1) / 2 * Ma_e ** 2 * (F1[:, 1]) ** 2), y1 / deltaB0, linewidth=1, color='g',
         linestyle='dashed')

ax1.plot(F2[:, 3], y2 / deltaB0, linewidth=1, color='r', label='$T_{w}/T_e$ = 8')
ax1.plot(F2[:, 3] + ((gamma - 1) / 2 * Ma_e ** 2 * (F2[:, 1]) ** 2), y2 / deltaB0, linewidth=1, color='r',
         linestyle='dashed')

ax0.set_xlabel('$u/V_e$')
ax0.set_ylabel('$y/\delta_{BO}$')
ax0.set_xlim(0, 1.2)
ax0.set_ylim(0, 20)
ax0.grid(linestyle='--')
ax0.tick_params(direction='in', length=8)
ax0.legend(loc='best', fontsize=16)


textstr1 = 'Solid: $T/T_e$'
textstr2 = 'Dashed: $T_0/T_{e}$'
ax1.text(4.0, 7, textstr1, fontsize=16, verticalalignment='top', horizontalalignment='right')
ax1.text(6.0, 18, textstr2, fontsize=16, verticalalignment='top', horizontalalignment='right')

if flag_ad == 1:
    strTemp = str('Adiabatic')
else:
    strTemp = str(r'$T_w/T_e=%1.1f$' % (g_w, ))
textstr = (r'$Ma_e=%2d$' % (Ma_e, ) + ', ' +
           r'$\gamma=%1.1f$' % (gamma, ) + ', ' +
           r'$Pr=%1.2f$' % (Pr, ) + ', ' +
           r'$\sigma=%.2f$' % (sigma, ) + ', ' +
           strTemp)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(9, 21.5, textstr, fontsize=14, verticalalignment='top', horizontalalignment='right', bbox=props)

ax1.set_xlabel('$T/T_e$ or $T_0/T_{e}$')
ax1.set_xlim(1, 8)
ax1.set_ylim(0, 20)
ax1.grid(linestyle='--')
ax1.tick_params(direction='in', length=8)

plt.savefig('./Figures/fig3.svg')
plt.show()


# Stanton number and Skin Friction coefficient calculation
# Friction Coefficient
Cf = (np.sqrt(2) / np.sqrt(Re)) * np.array([Cw_0 * F0[0, 2], Cw_1 * F1[0, 2], Cw_2 * F2[0, 2]])
St = [0,
      (1 / np.sqrt(2)) * (Cw_1 / Pr) * (T_e / (T_aw - Tw_1)) * F1[0, 4] / np.sqrt(Re),
      (1 / np.sqrt(2)) * (Cw_2 / Pr) * (T_e / (T_aw - Tw_2)) * F2[0, 4] / np.sqrt(Re)]

#%%
print('The Cf for the adiabatic case is {:.5f}'.format(Cf[0]))
print('The Cf for the isothermal case (Tw/Te = 4) is {:.5f}'.format(Cf[1]))
print('The Cf for the isothermal case (Tw/Te = 8) case is {:.5f}'.format(Cf[2]))

print('The St number for the adiabatic case is 0')
print('The St number for the isothermal case (Tw/Te = 4) is {:.5f}'.format(St[1]))
print('The St number for the isothermal case (Tw/Te = 8) is {:.5f}'.format(St[2]))



#%%




