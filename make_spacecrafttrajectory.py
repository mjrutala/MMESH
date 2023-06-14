#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:17:07 2023

@author: mrutala
"""

def MSWIM2Dlike():
    import datetime as dt
    import numpy as np
    import spiceypy as spice
    import pandas as pd
    
    import spacecraftdata as SpacecraftData
    
    #  Things that will be input parameters
    name = 'Ulysses'
    starttime = dt.datetime(1990, 1, 1)
    stoptime = dt.datetime(2010, 1, 1)
    latitude_cutoff = 10
    
    
    sc = SpacecraftData.SpacecraftData(name)
    sc.read_processeddata(starttime, stoptime)
    sc.find_state('SUN_INERTIAL', 'SUN')
    
    
    rlonlat = [spice.reclat(np.array(row[['xpos', 'ypos', 'zpos']])) for index, row in sc.data.iterrows()]
    rlonlat = np.array(rlonlat)
    
    criteria_index = np.where((rlonlat[:,0]/sc.au_to_km >= 4.5) 
                              & (rlonlat[:,0]/sc.au_to_km <= 5.5) 
                              & (rlonlat[:,2]*180/np.pi > -latitude_cutoff) 
                              & (rlonlat[:,2]*180/np.pi < latitude_cutoff))
    
    sc.data = sc.data.iloc[criteria_index]
    
    df_to_write = pd.DataFrame()
    df_to_write['year'] = [t.strftime('%Y') for t in sc.data.index]
    df_to_write['mo'] = [t.strftime('%m') for t in sc.data.index]
    df_to_write['dy'] = [t.strftime('%d') for t in sc.data.index]
    df_to_write['hr'] = [t.strftime('%H') for t in sc.data.index]
    df_to_write['mn'] = [t.strftime('%M') for t in sc.data.index]
    df_to_write['sc'] = [t.strftime('%S') for t in sc.data.index]
    df_to_write['msc'] = [t.strftime('%f') for t in sc.data.index]
    df_to_write['x'] = sc.data['xpos']/sc.au_to_km
    df_to_write['y'] = sc.data['ypos']/sc.au_to_km
    df_to_write['z'] = sc.data['zpos']/sc.au_to_km
    
    trajectory_filename = ''.join([sc.name.lower(), '_trajectory_',
                                   sc.data.index[0].strftime('%Y%m%d%H%M%S'),
                                   '-', 
                                   sc.data.index[-1].strftime('%Y%m%d%H%M%S'), 
                                   '.dat'])
    
    with open(trajectory_filename, "w") as file:
        file.write('MSWIM2D Trajectory File' + '\n')
        file.write(' Satellite: ' + sc.name.upper() + '\n')
        file.write(' Coordinate System: HGI' + '\n')
        file.write(' Units: AU' + '\n')
        file.write('\n')
        file.write('#COORD' + '\n')
        file.write('HGI' + '\n')
        file.write('\n')
        file.write('year mo dy hr mn sc msc x y z' + '\n')
        file.write('#START')
    
    df_to_write.to_csv(trajectory_filename, sep=' ', mode='a', header=False, index=False)
    
    