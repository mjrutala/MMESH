#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:44:58 2023

@author: mrutala
"""
import numpy as np

def get_JoyBowShock(p_ram=0.1, x=None, y=None, z=None):
    """
    One of x, y, and z should be set to a constant value,
    and one should be set to a numpy array. All are assumed in R_J (71492 km)
    It is HIGHLY reccomended that you choose the abcissa to be single-valued, 
    and allow the ordinate to be double-valued. This means solving for:
        x in terms of abcissa y
        x in terms of abcissa z
        z in terms of abcissa y

    Parameters
    ----------
    p_ram : TYPE, optional
        DESCRIPTION. The default is 0.1.
    x : TYPE, optional
        DESCRIPTION. The default is None.
    y : TYPE, optional
        DESCRIPTION. The default is None.
    z : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """     
    print(type(x), type(y), type(z))
    #  Coefficients as a function of ram pressure
    #  All assumed to be for JSS coords in R_J/120. units
    A = -1.107 + 1.591*p_ram**(-1/4)
    B = -0.566 - 0.812*p_ram**(-1/4)
    C =  0.048 - 0.059*p_ram**(-1/4)
    D =  0.077 - 0.038*p_ram
    E = -0.874 - 0.299*p_ram
    F = -0.055 + 0.124*p_ram
    
    #  Joy+ 2002 Model:
    #   z^2 = A + B*x + C*x^2 + D*y + E*y^2 + F*x*y 
    #  
    if x is None:
        y, z = y/120., z/120.
        aq = C
        bq = B + F*y
        cq = A + D*y + E*y**2 - z**2
    if y is None:
        x, z = x/120., z/120.
        aq = E
        bq = D + F*x
        cq = A + B*x + C*x**2 - z**2
    if z is None:
        x, y = x/120., y/120.
        aq = -1.0
        bq = 0.0
        cq = A + B*x + C*x**2 + D*y + E*y**2 + F*x*y
            
    ordinate_p = ((-bq + np.sqrt(bq**2 - 4*aq*cq))/(2*aq)) * 120.
    ordinate_m = ((-bq - np.sqrt(bq**2 - 4*aq*cq))/(2*aq)) * 120.   
    
    print(np.nanmin(ordinate_p), np.nanmax(ordinate_p))
    print(np.nanmin(ordinate_m), np.nanmax(ordinate_m))
       
    return ordinate_p, ordinate_m