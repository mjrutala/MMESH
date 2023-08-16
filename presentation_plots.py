#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 12:53:24 2023

@author: mrutala

Plot some useful plots for presentations; made for MIT Haystack visit on 20230817
"""
import datetime as dt

import SWData_Analysis as swda
#import DTW_Application as dtwa

starttime = dt.datetime(2016, 5, 15)
endtime = dt.datetime(2016, 7, 1)

_ = swda.plot_SingleTimeseries('u_mag', 'Juno', ['Tao', 'ENLIL', 'HUXt'], starttime, endtime)

_ = swda.plot_TaylorDiagram_Baseline('u_mag', 'Juno', ['Tao', 'ENLIL', 'HUXt'], starttime, endtime)
output1 = swda.plot_TaylorDiagram_ConstantTimeWarping('u_mag', 'Juno', ['Tao', 'ENLIL', 'HUXt'], starttime, endtime)


_ = swda.plot_SingleTimeseries('p_dyn', 'Juno', ['Tao', 'ENLIL'], starttime, endtime)

_ = swda.plot_TaylorDiagram_Baseline('p_dyn', 'Juno', ['Tao', 'ENLIL'], starttime, endtime)
output2 = swda.plot_TaylorDiagram_ConstantTimeWarping('p_dyn', 'Juno', ['Tao', 'ENLIL'], starttime, endtime)