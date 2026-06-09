#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:48:48 2026

@author: cowie
"""

import numpy as np
from scipy.optimize import root
import scipy.special as special
from scipy.special import expit  # numerically stable sigmoid


######################## Constants

electron_mass_cgs = 9.1094e-28
electron_charge_cgs = 4.803e-10
c_cgs = 3e10
c1 = 6.27e18
kB_cgs = 1.38e-16

######################## Pseudo-constants and helper functions

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

def k3_nu_func(p):
    return (2*c1)**(-(p+4)/34) * (1/(8*np.pi) * 4/3 * 11/6 * c6_func(p) )**(-1/17)

def K_E_func(p, Emin, Emax):
    if 1.99 < p < 2.01:
        return np.log(Emax/Emin)
    else:
        return 1/(2-p) * (Emax**(2-p) - Emin**(2-p))
    
def K_E_dens_func(p, Emin, Emax):
    if p == 1.0:
        return np.log(Emax/Emin)
    else:
        return 1/(1-p) * (Emax**(1-p) - Emin**(1-p))

#CHECK ALL FREQUENCY FORMS
def K_nu_func(p, nu_min, nu_max):
    if 1.99 < p < 2.01:
        return 0.5 * np.log(nu_max/nu_min)
    else:
        return 0.5 * c1**((p-2)/2) * 2/(2-p) * (nu_max**((2-p)/2) - nu_min**((2-p)/2))
    
def K_nu_dens_func(p, nu_min, nu_max):
    if p == 1.0:
        return 0.5 * np.log(nu_max/nu_min)
    else:
        return 0.5 * c1**((p-1)/2) * 2/(1-p) * (nu_max**((1-p)/2) - nu_min**((1-p)/2))
    
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

def nu_crit(electron_energy,B):
    return c1 * B * electron_energy**2


######################## Base quantities - energy form - checked against previous equations - to be checked against simulated spectra

#Total energy in the magnetic field and non-thermal particles (including protons) of the emitting region
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

#Size of (quasi-spherical) emitting region
def R_energy_form(flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, gamma_min, gamma_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation, log10=True):
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
    
    size = ( k3_e_func(p) * k1_func(p)**(-(2*(p+6))/(4*(1+2*(p+6)))) * K_E_func(p, Emin, Emax)**(1/(1+2*(p+6))) * nu1**(-1) * 
            dist_cm**(2*(p+6)/(1+2*(p+6))) * fnu1**((p+6)/(1+2*(p+6))) * doppler**(-(p+5)/(1+2*(p+6))) * (1+redshift)**(-(1+3*(p+6))/(1+2*(p+6))) * 
            (1+proton_energy_ratio)**(1/(1+2*(p+6))) * equip_deviation**(1/(1+2*(p+6))) ) 
    
    if log10:
        return np.log10(size)
    else:
        return size

#Magnetic field in emitting region
def B_energy_form(flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, gamma_min, gamma_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation, log10=True):
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
    
    B = (  k3_e_func(p)**(4) * k1_func(p)**(1/(1+2*(p+6))) * K_E_func(p, Emin, Emax)**(4/(1+2*(p+6))) * nu1 * 
         dist_cm**(-4/(1+2*(p+6))) * fnu1**(-2/(1+2*(p+6))) * doppler**(-(2*p+7)/(1+2*(p+6))) * 
         (1+redshift)**((1+2*(p+7))/(1+2*(p+6))) * (1+proton_energy_ratio)**(4/(1+2*(p+6))) * equip_deviation**(4/(1+2*(p+6))) )
    
    if log10:
        return np.log10(B)
    else:
        return B
    
######################## Base quantities - frequency form

#CHECK ALL FREQUENCY FORMS    
#Total energy in the magnetic field and non-thermal particles (including protons) of the emitting region
def E_frequency_form(flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, nu_min, nu_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation, log10=True):
    p = p_from_alpha(alpha_thin)
    
    tau_m = root(optical_depth_to_be_minimised, 1, args=(p,)).x[0]
    nu1 = nu1_from_numax(nu_obs_Hz, p, tau_m)
    
    flux_dens_peak_cgs = flux_dens_peak_mJy * 1e-26
    
    fnu1 = fnu1_from_fmax(flux_dens_peak_cgs, p, nu_obs_Hz, nu1)
    
    dist_cm = kpc_to_cm(dist_kpc)
    
    inclination_deg = np.arccos(cos_inclination) * 180/np.pi
    doppler = doppler_func(bulk_gamma, inclination_deg)
    
    
    deviation_factor = (6/17 * equip_deviation**(11/17) + 11/17 * equip_deviation**(-6/17))
    
    energy = ( 17/36 * k1_func(p)**(-10/17) * k3_nu_func(p)**(11) * K_nu_func(p, nu_min, nu_max)**(11/17) * 
              fnu1**(20/17) * dist_cm**(40/17) * nu1**((11*p - 56)/34) * doppler**(-(64+11*p)/34) * 
              (1+redshift)**((11*p + 96)/34) * (1+proton_energy_ratio)**(11/17) * deviation_factor )
    
    if log10:
        return np.log10(energy)
    else:
        return energy
    
def R_frequency_form(flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, nu_min, nu_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation, log10=True):
    p = p_from_alpha(alpha_thin)
    
    tau_m = root(optical_depth_to_be_minimised, 1, args=(p,)).x[0]
    nu1 = nu1_from_numax(nu_obs_Hz, p, tau_m)
    
    flux_dens_peak_cgs = flux_dens_peak_mJy * 1e-26
    
    fnu1 = fnu1_from_fmax(flux_dens_peak_cgs, p, nu_obs_Hz, nu1)
    
    dist_cm = kpc_to_cm(dist_kpc)
    
    inclination_deg = np.arccos(cos_inclination) * 180/np.pi
    doppler = doppler_func(bulk_gamma, inclination_deg)
    
    size = ( k1_func(p)**(-4/17) * k3_nu_func(p) * K_nu_func(p, nu_min, nu_max)**(1/17) * 
            fnu1**(8/17) * dist_cm**(16/17) * nu1**((p-36)/34) * doppler**(-(12+p)/34) * 
            (1+redshift)**((p-52)/34) * (1+proton_energy_ratio)**(1/17) * equip_deviation**(1/17) )
    
    if log10:
        return np.log10(size)
    else:
        return size
    
def B_frequency_form(flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, nu_min, nu_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation, log10=True):
    p = p_from_alpha(alpha_thin)
    
    tau_m = root(optical_depth_to_be_minimised, 1, args=(p,)).x[0]
    nu1 = nu1_from_numax(nu_obs_Hz, p, tau_m)
    
    flux_dens_peak_cgs = flux_dens_peak_mJy * 1e-26
    
    fnu1 = fnu1_from_fmax(flux_dens_peak_cgs, p, nu_obs_Hz, nu1)
    
    dist_cm = kpc_to_cm(dist_kpc)
    
    inclination_deg = np.arccos(cos_inclination) * 180/np.pi
    doppler = doppler_func(bulk_gamma, inclination_deg)
    
    B = ( k1_func(p)**(1/17) * k3_nu_func(p)**(4) * K_nu_func(p, nu_min, nu_max)**(4/17) * 
            fnu1**(-2/17) * dist_cm**(-4/17) * nu1**((2*p+13)/17) * doppler**(-(2*p+7)/17) * 
            (1+redshift)**((2*p+15)/17) * (1+proton_energy_ratio)**(4/17) * equip_deviation**(4/17) )
    
    if log10:
        return np.log10(B)
    else:
        return B

######################## Dervied quantities - energy form

#Normalisation for electron energy distribution power law
def N0_energy_form(flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, gamma_min, gamma_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation, log10=True):
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
    
    N0 = ( 11/(16*np.pi*(p+1)) * k3_e_func(p)**(8) * k1_func(p)**(2/(1+2*(p+6))) * K_E_func(p, Emin, Emax)**(-(5+2*p)/(1+2*(p+6))) * nu1**2 *
          dist_cm**(-8/(1+2*(p+6))) * fnu1**(-4/(1+2*(p+6))) * doppler**(-(4*p+14)/(1+2*(p+6))) * (1+redshift)**((2+4*(p+7))/(1+2*(p+6))) * 
          (1+proton_energy_ratio)**(-(5+2*p)/(1+2*(p+6))) * equip_deviation**(-(5+2*p)/(1+2*(p+6))) )
    
    if log10:
        return np.log10(N0)
    else:
        return N0
    
#NUmber density of non-thermal electrons
def ne_energy_form(flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, gamma_min, gamma_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation, log10=True):
    p = p_from_alpha(alpha_thin)
    Emin = gamma_to_energy(gamma_min)
    Emax = gamma_to_energy(gamma_max)
    
    ne =  N0_energy_form(flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, gamma_min, gamma_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation, log10=False) * K_E_dens_func(p, Emin, Emax)

    if log10:
        return np.log10(ne)
    else:
        return ne
    
#Total number of non-thermal electrons in emitting volume
def Ne_energy_form(flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, gamma_min, gamma_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation, log10=True):
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
    
    Ne = (  11/(12*(p+1)) * K_E_dens_func(p, Emin, Emax) * k3_e_func(p)**11 * k1_func(p)**(-(14+3*p)/(2*(1+2*(p+6)))) * K_E_func(p, Emin, Emax)**(-(2+2*p)/(1+2*(p+6))) * 
          nu1**(-1) * dist_cm**((4+6*(p+4))/(1+2*(p+6))) * fnu1**((2+3*(p+4))/(1+2*(p+6))) * doppler**(-(1+7*(p+4))/(1+2*(p+6))) * 
          (1+redshift)**(-(2+5*(p+5))/(1+2*(p+6))) * (1+proton_energy_ratio)**(-(2+2*p)/(1+2*(p+6))) * equip_deviation**(-(2+2*p)/(1+2*(p+6)))  )
    
    if log10:
        return np.log10(Ne)
    else:
        return Ne

#Brightness temperature of emitting region
def TB_energy_form(flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, gamma_min, gamma_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation, log10=True):
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
    
    const = c_cgs**2 / (2*np.pi*kB_cgs)
    
    TB = const * (  k3_e_func(p)**(-2) * k1_func(p)**((p+6)/(1+2*(p+6))) * K_E_func(p, Emin, Emax)**(-2/(1+2*(p+6))) * 
          dist_cm**(2/(1+2*(p+6))) * fnu1**(1/(1+2*(p+6))) * doppler**(-3/(1+2*(p+6))) * (1+redshift)**(-1/(1+2*(p+6))) * (1+proton_energy_ratio)**(-2/(1+2*(p+6))) * equip_deviation**(-2/(1+2*(p+6))) )
    
    if log10:
        return np.log10(TB)
    else:
        return TB
    
#################### Gamma min equations - energy form

#Function to be minimised numerically to find lower limit of gamma_min
def gamma_min_func_to_minimise(gamma_min, flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, gamma_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation):
    
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
    
    factor = (  c1**(-1) * k3_e_func(p)**(-4) * k1_func(p)**(-1/(1+2*(p+6))) * K_E_func(p, Emin, Emax)**(-4/(1+2*(p+6))) * 
              dist_cm**(4/(1+2*(p+6))) * fnu1**(2/(1+2*(p+6))) * doppler**(-6/(1+2*(p+6))) *  (1+redshift)**(-2/(1+2*(p+6))) * (1+proton_energy_ratio)**(-4/(1+2*(p+6))) * equip_deviation**(-4/(1+2*(p+6))) )
        
    
    return (electron_mass_cgs**2 * c_cgs**4 * gamma_min**2 - factor)


def gamma_min_func_to_minimise_x(x, flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, gamma_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation):
    
    gamma_min = gamma_max * expit(x)  # always 0 < gamma_min < gamma_max
    
    return gamma_min_func_to_minimise(
        gamma_min, flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin,
        gamma_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio,
        equip_deviation
    )

#Lower limit on gamma_min, gamma_min taken as argument but not used in calculation
#Fixed the self consistency problem - I wasn't calculating B with the correct gamma_min leading to the weirdness, will also of course confirm with final testing but it looks good now
def gamma_min_constraint(flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, gamma_min, gamma_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation, init=10, log10=True):
    
    gamma_min_upper_lim_x = root(gamma_min_func_to_minimise_x, 0.0, args=(flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, gamma_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation,)).x[0]
    
    gamma_min_upper_lim = gamma_max * expit(gamma_min_upper_lim_x)
    
    if log10:
        return np.log10(gamma_min_upper_lim)
    else:
        return gamma_min_upper_lim
    
######################## Derived quantities - frequency form

#Normalisation for electron energy distribution power law
def N0_frequency_form(flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, nu_min, nu_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation, log10=True):
    p = p_from_alpha(alpha_thin)
    
    tau_m = root(optical_depth_to_be_minimised, 1, args=(p,)).x[0]
    nu1 = nu1_from_numax(nu_obs_Hz, p, tau_m)
    
    flux_dens_peak_cgs = flux_dens_peak_mJy * 1e-26
    
    fnu1 = fnu1_from_fmax(flux_dens_peak_cgs, p, nu_obs_Hz, nu1)
    
    dist_cm = kpc_to_cm(dist_kpc)
    
    inclination_deg = np.arccos(cos_inclination) * 180/np.pi
    doppler = doppler_func(bulk_gamma, inclination_deg)
    
    N0 = ( 11/(48*np.pi) * (k1_func(p)**(1/17) * k3_nu_func(p)**(4) * fnu1**(-2/17) * dist_cm**(-4/17) * nu1**((2*p+13)/17) * doppler**(-(2*p+7)/17) * 
          (1+redshift)**((2*p+15)/17) )**((6-p)/2) * 
          (1+proton_energy_ratio)**(-(5+2*p)/17) * equip_deviation**(-(5+2*p)/17) * K_nu_func(p, nu_min, nu_max)**(-(5+2*p)/17) )
    
    if log10:
        return np.log10(N0)
    else:
        return N0
    
def ne_frequency_form(flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, nu_min, nu_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation, log10=True):
    p = p_from_alpha(alpha_thin)
    
    tau_m = root(optical_depth_to_be_minimised, 1, args=(p,)).x[0]
    nu1 = nu1_from_numax(nu_obs_Hz, p, tau_m)
    
    flux_dens_peak_cgs = flux_dens_peak_mJy * 1e-26
    
    fnu1 = fnu1_from_fmax(flux_dens_peak_cgs, p, nu_obs_Hz, nu1)
    
    dist_cm = kpc_to_cm(dist_kpc)
    
    inclination_deg = np.arccos(cos_inclination) * 180/np.pi
    doppler = doppler_func(bulk_gamma, inclination_deg)
    
    ne = ( 11/(48*np.pi) * K_nu_dens_func(p, nu_min, nu_max) * k1_func(p)**(5/34) * k3_nu_func(p)**(10) * K_nu_func(p, nu_min, nu_max)**(-7/17) * 
          fnu1**(-10/34) * dist_cm**(-20/34) * nu1**((10*p + 65)/34) * doppler**(-(10*p+35)/34) * (1+redshift)**((10*p+75)/34) * 
          (1+proton_energy_ratio)**(-7/17) * equip_deviation**(-7/17) )
    
    if log10:
        return np.log10(ne)
    else:
        return ne
    
def Ne_frequency_form(flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, nu_min, nu_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation, log10=True):
    p = p_from_alpha(alpha_thin)
    
    tau_m = root(optical_depth_to_be_minimised, 1, args=(p,)).x[0]
    nu1 = nu1_from_numax(nu_obs_Hz, p, tau_m)
    
    flux_dens_peak_cgs = flux_dens_peak_mJy * 1e-26
    
    fnu1 = fnu1_from_fmax(flux_dens_peak_cgs, p, nu_obs_Hz, nu1)
    
    dist_cm = kpc_to_cm(dist_kpc)
    
    inclination_deg = np.arccos(cos_inclination) * 180/np.pi
    doppler = doppler_func(bulk_gamma, inclination_deg)
    
    Ne = ( 11/36 * K_nu_dens_func(p, nu_min, nu_max) * k1_func(p)**(-19/34) * k3_nu_func(p)**(13) * K_nu_func(p, nu_min, nu_max)**(-4/17) * 
          fnu1**(19/17) * dist_cm**(38/17) * nu1**((13*p - 43)/34) * doppler**(-(13*p+71)) * (1+redshift)**((13*p-81)/34) * 
          (1+proton_energy_ratio)**(-4/17) * equip_deviation**(-4/17)  )
    
    if log10:
        return np.log10(Ne)
    else:
        return Ne
    
def TB_frequency_form(flux_dens_peak_mJy, nu_obs_Hz, dist_kpc, alpha_thin, nu_min, nu_max, bulk_gamma, cos_inclination, redshift, proton_energy_ratio, equip_deviation, log10=True):
    p = p_from_alpha(alpha_thin)
    
    tau_m = root(optical_depth_to_be_minimised, 1, args=(p,)).x[0]
    nu1 = nu1_from_numax(nu_obs_Hz, p, tau_m)
    
    flux_dens_peak_cgs = flux_dens_peak_mJy * 1e-26
    
    fnu1 = fnu1_from_fmax(flux_dens_peak_cgs, p, nu_obs_Hz, nu1)
    
    dist_cm = kpc_to_cm(dist_kpc)
    
    inclination_deg = np.arccos(cos_inclination) * 180/np.pi
    doppler = doppler_func(bulk_gamma, inclination_deg)
    
    const = c_cgs**2 / (2*np.pi*kB_cgs)
    
    TB = const * ( k1_func(p)**(8/17) * k3_nu_func(p)**(-2) * K_nu_func(p, nu_min, nu_max)**(-2/17) * fnu1**(1/17) * 
                  dist_cm**(2/17) ** nu1**((2-p)/17) * doppler**((p-5)/17) * (1+redshift)**((1-p)/17) * 
                  (1+proton_energy_ratio)**(-2/17) * equip_deviation**(-2/17) )
    
    if log10:
        return np.log10(TB)
    else:
        return TB

###### p = 2.0 case testing

# alpha_thin = -0.5
# p = p_from_alpha(alpha_thin)
# tau_m = root(optical_depth_to_be_minimised, 1, args=(p,)).x[0]
# nu1 = nu1_from_numax(1e9, p, tau_m)
# print(p)

# energy = E_energy_form(1, 1e9, 1, -0.5, 10, 100, 1, 0, 0, 0, 1)
# size = R_energy_form(1, 1e9, 1, -0.5, 10, 100, 1, 0, 0, 0, 1)
# B = B_energy_form(1, 1e9, 1, -0.5, 10, 100, 1, 0, 0, 0, 1, log10=False)

# energy_nu = E_frequency_form(1, 1e9, 1, -0.5, 1e8, 1e10, 1, 0, 0, 0, 1)
# size_nu = R_frequency_form(1, 1e9, 1, -0.5, 1e8, 1e10, 1, 0, 0, 0, 1)
# B_nu = B_frequency_form(1, 1e9, 1, -0.5, 1e8, 1e10, 1, 0, 0, 0, 1, log10=False)


# print(energy)
# print(energy_nu)

# print(size)
# print(size_nu)

# print(B)
# print(B_nu)

# N0 = N0_energy_form(1, 1e9, 1, -0.5, 10, 100, 1, 0, 0, 0, 1, log10=False)
# N0_nu = N0_frequency_form(1, 1e9, 1, -0.5, 1e8, 1e10, 1, 0, 0, 0, 1, log10=False)

# print(N0)
# print(N0_nu)

# Ee = N0*K_E_func(p, 10, 100)
# Eb = B**2 / (np.pi * 8)

# print(Eb/Ee)

# Ee = N0_nu*K_nu_func(p, 1e8, 1e10)
# Eb = B_nu**2 / (np.pi * 8)

# print(Eb/Ee)

# ne = ne_energy_form(1, 1e9, 1, -0.5, 10, 100, 1, 0, 0, 0, 1, log10=False)
# ne_nu = ne_frequency_form(1, 1e9, 1, -0.5, 1.55e8, 1.55e10, 1, 0, 0, 0, 1, log10=False)

# print(ne)
# print(ne_nu)

# ne_nu2 = np.log10(N0_nu * B_nu**((p-1)/2) * K_nu_dens_func(p, 1.55e8, 1.55e10))

# print(ne_nu2)

# Ne = Ne_energy_form(1, 1e9, 1, -0.5, 10, 100, 1, 0, 0, 0, 1, log10=False)
# Ne_nu = Ne_frequency_form(1, 1e9, 1, -0.5, 1.55e8, 1.55e10, 1, 0, 0, 0, 1, log10=False)

# print(Ne)
# print(Ne_nu)

# Ne_pred = 4/3 * np.pi * R_energy_form(1, 1e9, 1, -0.5, 10, 100, 1, 0, 0, 0, 1, log10=False)**3 * ne
# Ne_pred_nu = 4/3 * np.pi * R_frequency_form(1, 1e9, 1, -0.5, 1.55e8, 1.55e10, 1, 0, 0, 0, 1, log10=False)**3 * ne_nu

# print(Ne_pred)
# print(Ne_pred_nu)

# TB = TB_energy_form(1, 1e9, 1, -0.5, 10, 100, 1, 0, 0, 0, 1, log10=True)
# TB_nu = TB_frequency_form(1, 1e9, 1, -0.5, 1e8, 1e10, 1, 0, 0, 0, 1, log10=True)

# print(TB)
# print(TB_nu)

# TB_pred = np.log10(c_cgs**2 / (2*np.pi*kB_cgs) * fnu1_from_fmax(1e-26, p, 1e9, nu1) * kpc_to_cm(1)**2 * nu1**(-2) * R_energy_form(1, 1e9, 1, -0.5, 10, 100, 1, 0, 0, 0, 1, log10=False)**(-2))
# TB_pred_nu = np.log10(c_cgs**2 / (2*np.pi*kB_cgs) * fnu1_from_fmax(1e-26, p, 1e9, nu1) * kpc_to_cm(1)**2 * nu1**(-2) * R_frequency_form(1, 1e9, 1, -0.5, 1e8, 1e10, 1, 0, 0, 0, 1, log10=False)**(-2))

# print(TB_pred)
# print(TB_pred_nu)

###############################

# print(k3_nu_func(p))
# print(k3_e_func(p))
# print(K_E_func(p, 10, 100))
# print(K_nu_func(p, 1e8, 1e10))


# gamma_min = gamma_min_constraint(1, 1e9, 1, -0.5, 10, 100, 1, 0, 0, 0, 1, log10=False)

# print(B)
# print(gamma_min)

# print(np.log10((gamma_min*electron_mass_cgs*c_cgs**2)**2 * c1 * B))
# print(np.log10(nu1))

# print(np.log10(np.abs(nu1 - c1*B*(gamma_min*electron_mass_cgs*c_cgs**2)**2)))

