import numpy as np
import pandas as pd

def sound_speed(T, gamma=1.4, Rg=287):
    a = np.sqrt(T * gamma * Rg)
    return a

def gas_ideal_density(T, P, Rg=287):
    rho = P / (Rg * T)
    return rho

def total_entalphy(h, U):
    ho = h + U**2 / 2
    return ho

def total_temperature(Ma, T, gamma):
    To = (1 + (gamma - 1) * Ma**2 / 2) * T
    return To

def total_pressure(Ma, P, gamma):
    Po = ((1 + (gamma - 1) * Ma**2 / 2)**(gamma / (gamma - 1))) * P
    return Po

def total_density(Ma, rho, gamma):
    rho_o = ((1 + (gamma - 1) * Ma ** 2 / 2) ** (1 / (gamma - 1))) * rho
    return rho_o

def downstream_prop(ps_rel, **kwargs):  # ps_rel stands for post shock relations
    gas_dict = kwargs.copy()
    for k in gas_dict.keys():
        if k == 'Pressure': gas_dict[k].append(gas_dict[k][0] * ps_rel.p21)
        if k == 'Temperature': gas_dict[k].append(gas_dict[k][0] * ps_rel.T21)
        if k == 'Density': gas_dict[k].append(gas_dict[k][0] * ps_rel.rho21)
        if k == 'Total_Pressure': gas_dict[k].append(gas_dict[k][0] * ps_rel.Po21)
        if k == 'Speed': gas_dict[k].append(gas_dict[k][0] * ps_rel.U21)
        gas_df = pd.DataFrame.from_dict(gas_dict, orient='index')
    return gas_df
