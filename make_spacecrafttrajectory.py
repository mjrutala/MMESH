#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:17:07 2023

@author: mrutala
"""

def MSWIM2Dlike(spacecraft, starttime, stoptime):
    import datetime as dt
    import numpy as np
    import spiceypy as spice
    import pandas as pd
    
    import spacecraftdata as SpacecraftData
    
    
    sc = SpacecraftData.SpacecraftData(spacecraft)
    sc.read_processeddata(starttime, stoptime)
    sc.find_state('SUN_INERTIAL', 'SUN')
    
    r_range = np.array((4.5, 6.0)) 
    lat_range = np.array((-10, 10))
    sc.find_subset(coord1_range=r_range*sc.au_to_km, coord3_range=lat_range * np.pi / 180., transform='reclat')
    
    if len(sc.data) == 0:
        return_str = 'No spacecraft data within the given date range fall between {} and {} AU and between {} and {} deg. latitude. Returning...'
        print(return_str.format(str(r_range[0]), str(r_range[1]), str(lat_range[0]), str(lat_range[1])))       
        
        return()
    
    # rlonlat = list()
    # for index, row in sc.data.iterrows():
    #     print(row)
    #     print(type(row))
    #     dummy1 = row[['x_pos', 'y_pos', 'z_pos']]
    #     dummy2 = np.array(dummy1, dtype='float64')
    #     dummy3 = spice.reclat(dummy2)
    #     rlonlat.append(dummy3)
    rlonlat = [spice.reclat(np.array(row[['x_pos', 'y_pos', 'z_pos']], dtype='float64')) for index, row in sc.data.iterrows()]
    rlonlat = np.array(rlonlat)
    
    df_to_write = pd.DataFrame()
    df_to_write['year'] = [t.strftime('%Y') for t in sc.data.index]
    df_to_write['mo'] = [t.strftime('%m') for t in sc.data.index]
    df_to_write['dy'] = [t.strftime('%d') for t in sc.data.index]
    df_to_write['hr'] = [t.strftime('%H') for t in sc.data.index]
    df_to_write['mn'] = [t.strftime('%M') for t in sc.data.index]
    df_to_write['sc'] = [t.strftime('%S') for t in sc.data.index]
    df_to_write['msc'] = [t.strftime('%f')[:-3] for t in sc.data.index] # micros to millis
    df_to_write['x'] = [coord/sc.au_to_km for coord in sc.data['x_pos']]
    df_to_write['y'] = [coord/sc.au_to_km for coord in sc.data['y_pos']]
    df_to_write['z'] = [coord/sc.au_to_km for coord in sc.data['z_pos']]
    
    trajectory_filename = ''.join([sc.name.lower(), '_trajectory_',
                                   starttime.strftime('%Y%m%d%H%M%S'),
                                   '-', 
                                   stoptime.strftime('%Y%m%d%H%M%S'),
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
        file.write('#START' + '\n')
    
    df_to_write.to_csv(trajectory_filename, sep=' ', mode='a', header=False, index=False, float_format='%.3f')
    
    