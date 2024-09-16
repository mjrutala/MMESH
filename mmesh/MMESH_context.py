#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 13:17:24 2023

@author: mrutala
"""
import numpy as np
import datetime as dt
import pandas as pd
import spiceypy as spice

#   !!!! To set up file
kernel_paths = ['/Users/mrutala/SPICE/generic/metakernel_planetary.txt',
                '/Users/mrutala/SPICE/juno/metakernel_juno.txt',
                '/Users/mrutala/SPICE/ulysses/metakernel_ulysses.txt']
SolarRadioFlux_filepath = '/Users/mrutala/Data/Sun/DRAO/penticton_radio_flux.csv'
OMNI_filedir = '/Users/mrutala/Data/OMNI/LowRes/'


#   MMESH should read DataFrames in from CSV
#   This is the most flexible way to distribute the code, as it doesn't require
#   special routines for formatting, etc. 
#   And it should be safe to assume the end user can, at worst, copy and paste things into
#   a formatted spreadsheet and run from there


def read_SolarRadioFlux(starttime, stoptime, sample_rate='60Min'):
    column_headers = ('date', 'observed_flux', 'adjusted_flux',)
    srf_df = pd.read_csv(SolarRadioFlux_filepath,
                         header = 0, names = column_headers)
    srf_df['date'] = [dt.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S')
                                for d in srf_df['date']]
    srf_df = srf_df.set_index('date')
    srf_df = srf_df.resample(sample_rate, origin='start_day').mean().interpolate(method='linear')

    #   Also take an 13-rotation rolling mean
    srf_df['smooth_adjusted_flux'] = srf_df['adjusted_flux'].rolling(dt.timedelta(days=13*27), center=True, min_periods=13*27*24).mean()

    result = srf_df[(srf_df.index >= starttime) & (srf_df.index <= stoptime)]
    
    return result

def make_PlanetaryContext_CSV(target, starttime, stoptime, kernel_paths=kernel_paths, filepath=''):
    #   !!!! Add option for resolution?
    datetimes = np.arange(starttime, stoptime, dt.timedelta(hours=1)).astype(dt.datetime)
    
    #   Load SPICE
    for kernel_path in kernel_paths:
        spice.furnsh(kernel_path)
    
    ets = spice.datetime2et(datetimes)
    
    target_xyz, _ = spice.spkpos(target, ets, 'SUN_INERTIAL', 'NONE', 'SUN')
    target_rlonlat = np.array([spice.reclat(xyz) for xyz in target_xyz])
        
    earth_xyz, _ = spice.spkpos('EARTH BARYCENTER', ets, 'SUN_INERTIAL', 'NONE', 'SUN')
    earth_rlonlat = np.array([spice.reclat(xyz) for xyz in earth_xyz])
    
    spice.kclear()
    
    del_r_TSE = target_rlonlat[:,0] - earth_rlonlat[:,0]
    del_lon_TSE = (target_rlonlat[:,1] - earth_rlonlat[:,1])*180/np.pi
    del_lat_TSE = (target_rlonlat[:,2] - earth_rlonlat[:,2])*180/np.pi
    
    context_df = pd.DataFrame(index=pd.DatetimeIndex(datetimes))
    context_df.loc[:, 'target_sun_earth_r'] = np.abs(del_r_TSE)
    context_df.loc[:, 'target_sun_earth_lon'] = np.abs(np.abs(del_lon_TSE + 180) % 360 - 180)
    context_df.loc[:, 'target_sun_earth_lat'] = np.abs(np.abs(del_lat_TSE + 180) % 360 - 180)
    context_df.loc[:, 'solar_radio_flux'] = read_SolarRadioFlux(starttime, stoptime, sample_rate='60Min')['adjusted_flux']
    context_df.loc[:, 'smooth_solar_radio_flux'] = read_SolarRadioFlux(starttime, stoptime, sample_rate='60Min')['smooth_adjusted_flux']
    context_df.loc[:, 'recurrence_index'] = find_RecurrenceIndex_fromOMNI(starttime, stoptime, sample_rat='60Min', smooth=True)
    
    filename = '{}_{}-{}_Context.csv'.format(target.lower().capitalize(),
                                             starttime.strftime('%Y%m%d%H%M%S'),
                                             stoptime.strftime('%Y%m%d%H%M%S'))
    fullfilepath = filepath / filename
    context_df.to_csv(str(fullfilepath), index_label='datetime')
    return fullfilepath

def make_SpacecraftContext_CSV(target, starttime, stoptime, kernel_paths=kernel_paths, filepath=''):
    #   !!!! Add option for resolution?
    datetimes = np.arange(starttime, stoptime, dt.timedelta(hours=1)).astype(dt.datetime)
    
    #   Load SPICE
    for kernel_path in kernel_paths:
        spice.furnsh(kernel_path)
    
    ets = spice.datetime2et(datetimes)
    
    target_xyz, _ = spice.spkpos(target, ets, 'SUN_INERTIAL', 'NONE', 'SUN')
    target_rlonlat = np.array([spice.reclat(xyz) for xyz in target_xyz])
        
    earth_xyz, _ = spice.spkpos('EARTH BARYCENTER', ets, 'SUN_INERTIAL', 'NONE', 'SUN')
    earth_rlonlat = np.array([spice.reclat(xyz) for xyz in earth_xyz])
    
    spice.kclear()
    
    del_r_TSE = target_rlonlat[:,0] - earth_rlonlat[:,0]
    del_lon_TSE = (target_rlonlat[:,1] - earth_rlonlat[:,1])*180/np.pi
    del_lat_TSE = (target_rlonlat[:,2] - earth_rlonlat[:,2])*180/np.pi
    
    context_df = pd.DataFrame(index=pd.DatetimeIndex(datetimes))
    context_df.loc[:, 'target_sun_earth_r'] = np.abs(del_r_TSE)
    context_df.loc[:, 'target_sun_earth_lon'] = np.abs(np.abs(del_lon_TSE + 180) % 360 - 180)
    context_df.loc[:, 'target_sun_earth_lat'] = np.abs(np.abs(del_lat_TSE + 180) % 360 - 180)
    context_df.loc[:, 'solar_radio_flux'] = read_SolarRadioFlux(starttime, stoptime, sample_rate='60Min')['adjusted_flux']
    context_df.loc[:, 'smooth_solar_radio_flux'] = read_SolarRadioFlux(starttime, stoptime, sample_rate='60Min')['smooth_adjusted_flux']
    context_df.loc[:, 'recurrence_index'] = find_RecurrenceIndex_fromOMNI(starttime, stoptime, sample_rate='60Min', smooth=True)
    
    
    filename = '{}_{}-{}_Context.csv'.format(target.lower().capitalize(),
                                             starttime.strftime('%Y%m%d%H%M%S'),
                                             stoptime.strftime('%Y%m%d%H%M%S'))
    fullfilepath = filepath / filename
    context_df.to_csv(str(fullfilepath), index_label='datetime')
    return fullfilepath

def find_RecurrenceIndex_fromOMNI(starttime, stoptime, smooth=True, sample_rate='60Min'):
    import glob
    import datetime
    
    filepaths = glob.glob(OMNI_filedir + 'omni_m????.dat')
    
    filepaths = sorted(filepaths)
    
    column_names = ['year', 'ddoy', 'hour', 
                    'lat_HGI', 'lon_HGI', 
                    'Br', 'Bt', 'Bn', 'B_avg', 
                    'U_sw', 'U_elevation_angle', 'U_azimuth_angle', 
                    'n_proton', 
                    'T']
    df_list = []
    for filepath in filepaths:
        df = pd.read_csv(filepath, sep='\s+', names=column_names)
        df_list.append(df)
        
    df = pd.concat(df_list, ignore_index=True)
    datetime_strings = ["{:04g}-{:03g}-{:02g}".format(*row[['year', 'ddoy', 'hour']].values) for _, row in df.iterrows()]
    df.index = [datetime.datetime.strptime(d, "%Y-%j-%H") for d in datetime_strings]
    
    df['U_sw_shifted'] = df['U_sw'].shift(-27, freq='D')
    
    #   Calculating the recurrence index only requires the bulk flow
    #   speed of the solar wind plasma
    u_df = df.loc[:, ['U_sw', 'U_sw_shifted']]
    
    #   Calculate the corr. coeff. matrix for each pair
    corr_df = u_df.rolling(datetime.timedelta(days=27), center=True, min_periods=27*24).corr(pairwise=True)
    
    #   Isolate the corr. coeff. between U_sw and U_sw_shifted
    corr_df = corr_df.xs('U_sw', axis='index', level=1)['U_sw_shifted'].to_frame(name='corr')
    
    #   Also take an 13-rotation rolling mean
    corr_df['smooth_corr'] = corr_df.rolling(datetime.timedelta(days=13*27), center=True, min_periods=13*27*24).mean()
    
    #   Subset both to lie between starttime (closed) and stoptime (open)
    corr_df = corr_df.query('@starttime <= index < @stoptime')
    
    result = corr_df['corr'].to_numpy()
    if smooth:
        result = corr_df['smooth_corr'].to_numpy()
    
    return result
    