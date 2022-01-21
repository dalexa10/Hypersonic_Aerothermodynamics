import math

from hyper_class.func import *
from hyper_class.post_shock import *
from hyper_class.post_fan import *
from ambiance import Atmosphere

#TODO Implement Jump Conditions in the Hypersonics Limit for both shock waves and expansion fans
#TODO Implement pressure coefficient of blunt bodies
#TODO Implement Viscous BL formulation (Prandtl)

if __name__ == '__main__':

    # Ambient Data
    atm = Atmosphere(29000)
    T_inf = atm.temperature[0]
    P_inf = atm.pressure[0]
    rho_inf = atm.density[0]
    a = sound_speed(T_inf)
    gamma = 1.4

    # Incoming Flow
    Ma1 = 9.6
    theta_shock = math.radians(13)
    theta_fan = math.radians(3)
    theta_fan2 = math.radians(22)

    # Post Shock Analysis and Initialization
    beta = beta_calculation(Ma1, theta_shock)
    Ma1n = Mach1_normal(Ma1, beta)

    # Thermodynamic Properties Downstream (State 2)
    Gas_preshock = {'Temperature': [T_inf],
                    'Pressure': [P_inf],
                    'Density': [rho_inf],
                    'Mach': [Ma1]}
    shock1_rel = Post_Shock(Ma1n, gamma)
    Gas2 = downstream_prop(shock1_rel, **Gas_preshock)
    Gas2.loc['Mach', 1] = Mach2(shock1_rel.Ma2, beta, theta_shock)

    # Post Fan Analysis
    Gas_prefan = {'Temperature': [T_inf],
                  'Pressure': [P_inf]}
    fan1_rel = Post_Fan(Ma1, theta_fan, gamma)
    Gas3 = downstream_prop(fan1_rel, **Gas_prefan)

    # Post Shock = Fan Analysis
    Gas_prefan2 = {'Temperature': [Gas2.loc['Temperature', 1]],
                    'Pressure': [Gas2.loc['Pressure', 1]],
                    'Mach': [Gas2.loc['Mach', 1]]}

    fan2_rel = Post_Fan(Gas2.loc['Mach', 1], theta_fan2, gamma)
    Gas4 = downstream_prop(fan2_rel, **Gas_prefan2)
    Gas4.loc['Mach', 1] = fan2_rel.Ma2

