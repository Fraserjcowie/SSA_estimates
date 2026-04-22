#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:48:48 2026

@author: cowie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import scipy.special as special
import yaml

from mc_tools import mc_uncertainty, plot_mc_diagnostics


electron_mass_cgs = 9.1094e-28
electron_charge_cgs = 4.803e-10
c_cgs = 3e10
c1 = 6.27e18



#Parameters required are F_nu1, dist, nu1, doppler, redshift, protons, deviation, p, cutoffs


def c5_func(p):
    prefactor = np.sqrt(3)/(16*np.pi) * electron_charge_cgs**3/(electron_mass_cgs * c_cgs**2)
    gammas = (p+7/3)/(p+1) * special.gamma((3*p - 1)/12) * special.gamma((3*p+7)/12)
    return prefactor*gammas

def c6_func(p):
    prefactor = np.sqrt(3) * np.pi/72 *  electron_charge_cgs * electron_mass_cgs**5 * c_cgs**10 
    gammas = (p+10/3) * special.gamma((3*p+2)/12) * special.gamma((3*p+10)/12)
    return prefactor*gammas

def k1_func(p):
    return (np.pi * c5_func(p)/c6_func(p))**2 * (2*c1)**(-5) * (1-np.exp(-1))**2

def k3_e_func(p):
    return (11/(2*(p+1)) * 1/(8*np.pi) * 4/3 * c6_func(p))**(-1/(1+2*(p+6))) * (2*c1)**(-(p+4)/(2+4*(p+6)))

def K_E_func(p, Emin, Emax):
    if p == 2.0:
        return np.log(Emax/Emin)
    else:
        return 1/(2-p) * (Emax**(2-p) - Emin**(2-p))
    
def p_from_alpha(alpha_thin):
    return 1-2*alpha_thin

def gamma_to_beta(gamma):
    return np.sqrt(1-1/gamma**2)

def doppler_func(gamma, inc):
    beta = gamma_to_beta(gamma)
    return (gamma*(1-beta*np.cos(np.pi/180 * inc)))**(-1)

def optical_depth_to_be_minimised(tau, p):
    return np.exp(tau) - 1 - (p+4)/5 * tau

def nu1_from_numax(numax, p, tau_max):
    return numax * tau_max**(2/(p+4))

def fnu1_from_fmax(fmax, p, nu_max, nu1):
    factor = (1-np.exp(-1))/((nu_max/nu1)**(5/2) * (1-np.exp(-(nu_max/nu1)**(-(p+4)/2))))
    return fmax * factor

def gamma_to_energy(gamma):
    return electron_mass_cgs * c_cgs**2 * gamma

def kpc_to_cm(dist_kpc):
    return dist_kpc * 3.086e21


########################

def E_energy_form(flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, gamma_min, gamma_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation, log10=True):
    p = p_from_alpha(alpha_thin)
    Emin = gamma_to_energy(gamma_min)
    Emax = gamma_to_energy(gamma_max)
    
    tau_m = root(optical_depth_to_be_minimised, 1, args=(p,)).x[0]
    nu1 = nu1_from_numax(nu_obs_Hz, p, tau_m)
    
    flux_dens_peak_cgs = flux_dens_peak_mJy * 1e-26
    
    fnu1 = fnu1_from_fmax(flux_dens_peak_cgs, p, nu_obs_Hz, nu1)
    
    dist_cm = kpc_to_cm(dist_kpc)
    
    inclination_deg = np.arccos(cos_inclination) * 180/np.pi
    doppler = doppler_func(bulk_gamma, inclination_deg)
    
    
    deviation_factor = (2*(1+p))/(1+2*(p+6)) * equip_deviation**(11/(1+2*(p+6))) + 11/(1+2*(p+6)) * equip_deviation**(-(2*(1+p))/(1+2*(p+6)))
    
    energy = ( (1+2*(p+6))/(2*(p+1)) * 4/3 * 1/8 * k3_e_func(p)**11 * k1_func(p)**(-(3*p+14)/(2*(1+2*(p+6)))) * K_E_func(p, Emin, Emax)**(11/(1+2*(p+6))) * 
            nu1**(-1) * dist_cm**((4+6*(p+4))/(1+2*(p+6))) * fnu1**((2+3*(p+4))/(1+2*(p+6))) * doppler**(-(1+7*(p+4))/(1+2*(p+6))) * (1+redshift)**(-(2+5*(p+5))) * 
            (1+proton_energy_ratio)**(11/(1+2*(p+6))) * deviation_factor)
    
    if log10:
        return np.log10(energy)
    else:
        return energy










########################


with open("/Users/cowie/Documents/DPhil/ssa_inhomogeneous/SSA_estimates/inputs.yaml") as f:
    params = yaml.safe_load(f)



res = mc_uncertainty(E_energy_form, params, n=1000, seed=100, ci_level=0.68)

plot_mc_diagnostics(res, params)



energy = E_energy_form(10, 1e9, 1, -0.5, 10, 100, 1, 0, 0, 0, 1)

print(energy)






