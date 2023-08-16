#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 12:53:24 2023

@author: mrutala

Plot some useful plots for presentations; made for MIT Haystack visit on 20230817
"""
import datetime as dt
import numpy as np

import SWData_Analysis as swda
#import DTW_Application as dtwa

starttime = dt.datetime(2016, 5, 10)
endtime = dt.datetime(2016, 7, 1)

_ = swda.plot_SingleTimeseries('u_mag', 'Juno', ['Tao', 'ENLIL', 'HUXt'], starttime, endtime)

u_mag_baseline = swda.plot_TaylorDiagram_Baseline('u_mag', 'Juno', ['Tao', 'ENLIL', 'HUXt'], starttime, endtime)
u_mag_shifts = swda.plot_TaylorDiagram_ConstantTimeWarping('u_mag', 'Juno', ['Tao', 'ENLIL', 'HUXt'], starttime, endtime)
print('Baseline correlation between Tao and Juno speeds: ' + str(u_mag_baseline['Tao']['r'].iloc[0]))
print('Max correlation between Tao and Juno speeds for constant shifts: ' + str(np.max(u_mag_shifts['Tao']['r'])))
print('Baseline correlation between Enlil and Juno speeds: ' + str(u_mag_baseline['ENLIL']['r'].iloc[0]))
print('Max correlation between Enlil and Juno speeds for constant shifts: ' + str(np.max(u_mag_shifts['ENLIL']['r'])))
print('Baseline correlation between HUXt and Juno speeds: ' + str(u_mag_baseline['HUXt']['r'].iloc[0]))
print('Max correlation between HUXt and Juno speeds for constant shifts: ' + str(np.max(u_mag_shifts['HUXt']['r'])))

_ = swda.plot_SingleTimeseries('p_dyn', 'Juno', ['Tao', 'ENLIL'], starttime, endtime)

p_dyn_baseline = swda.plot_TaylorDiagram_Baseline('p_dyn', 'Juno', ['Tao', 'ENLIL'], starttime, endtime)
p_dyn_shifts = swda.plot_TaylorDiagram_ConstantTimeWarping('p_dyn', 'Juno', ['Tao', 'ENLIL'], starttime, endtime)
print('Baseline correlation between Tao and Juno pressures: ' + str(p_dyn_baseline['Tao']['r'].iloc[0]))
print('Max correlation between Tao and Juno pressures for constant shifts: ' + str(np.max(p_dyn_shifts['Tao']['r'])))
print('Baseline correlation between Enlil and Juno pressures: ' + str(p_dyn_baseline['ENLIL']['r'].iloc[0]))
print('Max correlation between Enlil and Juno pressures for constant shifts: ' + str(np.max(p_dyn_shifts['ENLIL']['r'])))