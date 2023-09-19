#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:26:03 2023

@author: mrutala
"""

import datetime as dt
import pandas as pd
import numpy as np

import spacecraftdata
import read_SWModel
import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
import datetime
import os

import PlottingConstants as pc

r_range = [4.9, 5.5]
lat_range = [-6.1, 6.1]


def run_HUXtAtSpacecraft(spacecraft_names):
    import copy
    import matplotlib.pyplot as plt
    
    import spiceypy as spice
    
    import spacecraftdata as SpacecraftData
    
    import sys
    sys.path.append('/Users/mrutala/projects/HUXt-DIAS/code/')
    import huxt as H
    import huxt_analysis as HA
    import huxt_inputs as Hin
    
    # =============================================================================
    # Find
    # =============================================================================
    reference_frame = 'SUN_INERTIAL' # HGI  #'ECLIPJ2000'
    observer = 'SUN'
    
    spacecraft_list = []
    for sc_name in spacecraft_names:
        spacecraft_list.append(SpacecraftData.SpacecraftData(sc_name))
    
    
    for sc in spacecraft_list:
        sc.find_lifetime()
        
        sc.read_processeddata(sc.starttime, sc.stoptime)
        sc.find_state(reference_frame, observer)
        sc.find_subset(coord1_range=np.array(r_range)*sc.au_to_km, 
                       coord3_range=np.array(lat_range)*np.pi/180., 
                       transform='reclat')
        
        #  This bit splits the spacecraft data up into chunks to plot
        sc_deltas = sc.data.index.to_series().diff()
        gaps_indx = np.where(sc_deltas > dt.timedelta(days=30))[0]
        start_indx = np.insert(gaps_indx, 0, 0)
        stop_indx = np.append(gaps_indx, len(sc.data.index))-1
        
        print('Start and stop dates for ' + sc.name)
        for first, final in zip(start_indx, stop_indx):
            print(sc.data.index[first].strftime('%Y-%m-%d') + 
                  ' -- ' + 
                  sc.data.index[final].strftime('%Y-%m-%d'))
            print(str((sc.data.index[final] - sc.data.index[first]).total_seconds()/3600.) + ' total hours')
            rlonlat = [spice.reclat(np.array(row[['x_pos', 'y_pos', 'z_pos']], dtype='float64')) for indx, row in sc.data.iterrows()]
            rlonlat = np.array(rlonlat).T
            print('Radial range of: ' + str(np.min(rlonlat[0,:])/sc.au_to_km) +
                  ' - ' + str(np.max(rlonlat[0,:])/sc.au_to_km) + ' AU')
            print('Heliolatitude range of: ' + str(np.min(rlonlat[2,:])*180/np.pi) +
                  ' - ' + str(np.max(rlonlat[2,:])*180/np.pi) + 'deg.')
            print('------------------------------------------')      

def simpleHUXtRun(spacecraft_name, year):
    import time as benchmark_time
    
    import sys
    sys.path.append('/Users/mrutala/projects/HUXt-DIAS/code/')
    import huxt as H
    import huxt_analysis as HA
    import huxt_inputs as Hin
    
    runstart = dt.datetime(2016, 1, 1)
    runend = dt.datetime(2016, 2, 1)
    simtime = (runend-runstart).days * u.day
    r_min = 215 *u.solRad

    benchmark_starttime = benchmark_time.time()

    #download and process the OMNI data
    time, vcarr, bcarr = Hin.generate_vCarr_from_OMNI(runstart, runend)

    #!!!! Get spacecraft longitudes during this time span
    #  Or split up the time span dynamically so each HUXt run only needs to be ~2-4ish degrees lon.
    #  Then look into how sampling in 3D works (i.e., with latitude) -- hpopefully theres another easy sampler like sc_series...
    #  Then move this to read_model? Maybe with a renaming as well (Model class? models.py? ...)

    #set up the model, with (optional) time-dependent bpol boundary conditions
    model = Hin.set_time_dependent_boundary(vcarr, time, runstart, simtime, 
                                            r_min=r_min, r_max=1290*u.solRad, dt_scale=2.0, latitude=0*u.deg,
                                            bgrid_Carr = bcarr, lon_start=68*u.deg, lon_stop=72*u.deg, frame='sidereal')


    model.solve([])
    
    benchmark_totaltime = benchmark_time.time() - benchmark_starttime
    print('Time elapsed in solving the model: {}'.format(benchmark_totaltime))
    
    HA.plot(model, (0/4.)*simtime)
    HA.plot(model, (1/4.)*simtime)
    HA.plot(model, (2/4.)*simtime)
    HA.plot(model, (3/4.)*simtime)
    HA.plot(model, (4/4.)*simtime)

    sc_series = HA.get_observer_timeseries(model, observer=spacecraft_name)
    
    filename = spacecraft_name + '_' + str(year) + '_' + 'HUXt.csv'
    print(filename)
    sc_series.to_csv(filename)
    
    return(sc_series)


def get_SpacecraftLifetimes():
    import spacecraftdata as SpacecraftData
    
    # =========================================================================
    #   Would-be inputs
    # =========================================================================
    spacecraft_names = ['Juno', 'Ulysses', 'Voyager 1', 'Voyager 2', 'Pioneer 10', 'Pioneer 11']
    
    # =========================================================================
    #   Find when Spacecraft was within lat_range of the solar equator, 
    #   and within r_range from the sun
    # =========================================================================
    reference_frame = 'SUN_INERTIAL' # HGI  #'ECLIPJ2000'
    observer = 'SUN'

    # =========================================================================
    #   Total spacecraft lifetimes, hardcoded, from launch to loss of contact
    # =========================================================================
    spacecraft_lifetimes = {'Juno':         (dt.datetime(2011, 8, 5), 
                                             dt.datetime.now()),
                            'Ulysses':      (dt.datetime(1990, 10, 6), 
                                             dt.datetime(2009, 6, 30)),
                            'Voyager 1':    (dt.datetime(1977, 9, 1), 
                                             dt.datetime.now()),
                            'Voyager 2':    (dt.datetime(1977, 8, 20), 
                                             dt.datetime.now()),
                            'Pioneer 10':   (dt.datetime(1972, 3, 3),
                                             dt.datetime(2003, 1, 23)),
                            'Pioneer 11':   (dt.datetime(1973, 4, 6),
                                             dt.datetime(1995, 11, 24))}

    spacecraft_list = []
    times = {}
    for name in spacecraft_names:
        #  Initialize object
        sc_reltosun = SpacecraftData.SpacecraftData(name)
        
        #  Find the SPICE kernel coverage
        sc_reltosun.find_MetaKernelCoverage()
        
        #  Choose a start and stop time such that there is SPICE coverage
        if (spacecraft_lifetimes[name][0] > sc_reltosun.starttime):
            starttime = spacecraft_lifetimes[name][0]
        else:
            starttime = sc_reltosun.starttime
            
        if (spacecraft_lifetimes[name][1] < sc_reltosun.stoptime):
            stoptime = spacecraft_lifetimes[name][1]
        else:
            stoptime = sc_reltosun.stoptime
        
        #  Make a timeseries
        # sc_reltosun.make_timeseries(starttime, stoptime)
        
        # #  Check against the location requirements
        # sc_reltosun.find_state(reference_frame, observer)
        # sc_reltosun.find_subset(coord1_range=np.array(r_range)*sc_reltosun.au_to_km,
        #                 coord3_range=np.array(lat_range)*np.pi/180., 
        #                 transform='reclat')
         
        # spacecraft_list.append(sc_reltosun)
        times[name] = (starttime, stoptime)
        
    return times

def get_SpacecraftDataCoverage():
    import copy
    import spacecraftdata as SpacecraftData
    
    # =============================================================================
    #   Would-be inputs
    # =============================================================================
    spacecraft_names = ['Juno', 'Ulysses', 'Voyager 1', 'Voyager 2', 'Pioneer 10', 'Pioneer 11']
    
    # =============================================================================
    #   Find when Spacecraft was within lat_range of the solar equator, 
    #   and within r_range from the sun
    # =============================================================================
    reference_frame = 'SUN_INERTIAL' # HGI  #'ECLIPJ2000'
    observer = 'SUN'

    spacecraft_list = []
    times = {}
    for name in spacecraft_names:
        sc_reltosun = SpacecraftData.SpacecraftData(name)
        sc_reltosun.read_processeddata(everything=True)
        sc_reltosun.find_state(reference_frame, observer)
        sc_reltosun.find_subset(coord1_range=np.array(r_range)*sc_reltosun.au_to_km, # 4.4 - 6
                        coord3_range=np.array(lat_range)*np.pi/180., 
                        transform='reclat')
        
        # sc_reltojup = copy.deepcopy(sc_reltosun)
        
        # sc_reltosun.find_state('SUN_INTERTIAL', 'SUN')
        # sc.find_subset(coord1_range=np.array(r_range)*sc.au_to_km, # 4.4 - 6
        #                coord3_range=np.array(lat_range)*np.pi/180., 
        #                transform='reclat')
        
        # sc_reltojup.find_state('ECLIPJ2000', 'JUPITER')
        # sc.find_subset(coord1_range=np.array([])*sc.au_to_km, # 4.4 - 6
        #                coord3_range=np.array(lat_range)*np.pi/180., 
        #                transform='reclat')
        
        spacecraft_list.append(sc_reltosun)
        times[name] = (sc_reltosun.data.index[0].to_pydatetime(), 
                       sc_reltosun.data.index[-1].to_pydatetime())
        
    return times
    
def plot_spacecraftcoverage_solarcycle(spacecraft_lifetimes):
    import copy
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.collections as collections
    import matplotlib.dates as mdates
    
    #import datetime as dt
    #import numpy as np
    
    #import pandas as pd
    import spiceypy as spice
    
    #import sys
    #sys.path.append('/Users/mrutala/projects/SolarWindEM/')
    #import read_SWData
    import spacecraftdata as SpacecraftData
    
    # =============================================================================
    #   Would-be inputs
    # =============================================================================
    spacecraft_names = ['Juno', 'Ulysses', 'Voyager 1', 'Voyager 2', 'Pioneer 10', 'Pioneer 11']
    
        
    # =============================================================================
    # Find when Ulysses was within +/- 30 deg. of the ecliptic, and near 5 AU
    # =============================================================================
    reference_frame = 'SUN_INERTIAL' # HGI  #'ECLIPJ2000'
    observer = 'SUN'

    rj_to_km = 71492.
    
    solarwind_range = [120, np.inf]
    magnetopause_r = [70,120]  #  Feng+ 2023
    #magnetopause_lon = [90, 270]  #  lon is angle measured from x through y
    
    sw_trajectory_list = []
    mp_trajectory_list = []
    for name in spacecraft_names:
        
        sc_sw = SpacecraftData.SpacecraftData(name)
        sc_sw.make_timeseries(*spacecraft_lifetimes[sc_sw.name])
        sc_sw.find_state('SUN_INERTIAL', 'SUN')
        sc_sw.find_subset(coord1_range=np.array(r_range)*sc_sw.au_to_km,
                          coord3_range=np.array(lat_range)*np.pi/180., 
                          transform='reclat')
        
        #   We already limited to the solar range
        #   So now calculate position rel. to Jupiter for each of the remaining
        #   times, and make another cut
        sc_sw.find_state('IAU_JUPITER', 'JUPITER_BARYCENTER')
        sc_sw.find_subset(coord1_range=np.array(solarwind_range)*rj_to_km,
                          transform='reclat')
        
        sw_trajectory_list.append(sc_sw)
        
        #  Do it all again, but for a different cutoff:
        sc_mp = SpacecraftData.SpacecraftData(name)
        sc_mp.make_timeseries(*spacecraft_lifetimes[sc_mp.name])
        sc_mp.find_state('SUN_INERTIAL', 'SUN')
        sc_mp.find_subset(coord1_range=np.array(r_range)*sc_mp.au_to_km,
                          coord3_range=np.array(lat_range)*np.pi/180., 
                          transform='reclat')
        
        #   
        sc_mp.find_state('IAU_JUPITER', 'JUPITER_BARYCENTER')
        sc_mp.find_subset(coord1_range=np.array(magnetopause_r)*rj_to_km,
                          #coord3_range=np.array(magnetopause_lon)*np.pi/180.,
                          transform='reclat')
        
        mp_trajectory_list.append(sc_mp)
        
        fig, ax = plt.subplots()
        ax.scatter(sc_sw.data.index, sc_sw.data.index, s=0.1)
        ax.scatter(sc_mp.data.index, sc_mp.data.index, s=0.1)
        plt.show()
    
        #return sc_sw, sc_mp
    box_height = 0.8
    box_centers = {'Pioneer 10': 1,
                   'Pioneer 11': 2,
                   'Voyager 1': 3, 
                   'Voyager 2': 4, 
                   'Ulysses'  : 5, 
                   'Juno'     : 6}
    
    # =============================================================================
    # F10.4 Radio flux from the sun
    # =============================================================================
    column_headers = ('date', 'observed_flux', 'adjusted_flux',)
    solar_radio_flux = pd.read_csv('/Users/mrutala/Data/Sun/DRAO/penticton_radio_flux.csv',
                                   header = 0, names = column_headers)
    solar_radio_flux['date'] = [dt.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S')
                                for d in solar_radio_flux['date']]

    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        fig, axs = plt.subplots(nrows=2, height_ratios=(1,2), figsize=(6,4.5), sharex='col')
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
        axs[1].plot(solar_radio_flux['date'], solar_radio_flux['observed_flux'], 
                    marker='.', color='gray', markersize=2, linestyle='None')
        
        for sc in sw_trajectory_list:
            #  This bit splits the spacecraft data up into chunks to plot
            sc_deltas = sc.data.index.to_series().diff()
            gaps_indx = np.where(sc_deltas > dt.timedelta(days=5))[0]
            
            start_indx = np.insert(gaps_indx, 0, 0)
            stop_indx = np.append(gaps_indx, len(sc.data.index))-1
            
            print('Start and stop dates for ' + sc.name)
            for first, final in zip(start_indx, stop_indx):
                print(sc.data.index[first].strftime('%Y-%m-%d') + 
                      ' -- ' + 
                      sc.data.index[final].strftime('%Y-%m-%d'))
                print(str((sc.data.index[final] - sc.data.index[first]).total_seconds()/3600.) + ' total hours')
                rlonlat = [spice.reclat(np.array(row[['x_pos', 'y_pos', 'z_pos']], dtype='float64')) for indx, row in sc.data.iterrows()]
                rlonlat = np.array(rlonlat).T
                print('Radial range of: ' + str(np.min(rlonlat[0,:])/sc.au_to_km) +
                      ' - ' + str(np.max(rlonlat[0,:])/sc.au_to_km) + ' AU')
                print('Heliolatitude range of: ' + str(np.min(rlonlat[2,:])*180/np.pi) +
                      ' - ' + str(np.max(rlonlat[2,:])*180/np.pi) + 'deg.')
            print('------------------------------------------')
            
            #  If the spacecraft data is split into chunks, we don't want to plot
            #  a continuous line-- so use a collection of patches
            rectangle_list = []
            for x0, xf in zip(start_indx, stop_indx):
                
                ll_corner = (mdates.date2num(sc.data.index[x0]), box_centers[sc.name]-box_height/2.)
                
                width = mdates.date2num(sc.data.index[xf] + dt.timedelta(days=1)) - mdates.date2num(sc.data.index[x0])
                
                rectangle_list.append(patches.Rectangle(ll_corner, width, box_height))
                
                
            
            axs[0].add_collection(collections.PatchCollection(rectangle_list, facecolors=pc.spacecraft_colors[sc.name]))
            
           
        for sc in mp_trajectory_list:
            #  This bit splits the spacecraft data up into chunks to plot
            sc_deltas = sc.data.index.to_series().diff()
            gaps_indx = np.where(sc_deltas > dt.timedelta(days=5))[0]
            
            start_indx = np.insert(gaps_indx, 0, 0)
            stop_indx = np.append(gaps_indx, len(sc.data.index))-1
            
            print('Start and stop dates for ' + sc.name)
            for first, final in zip(start_indx, stop_indx):
                print(sc.data.index[first].strftime('%Y-%m-%d') + 
                      ' -- ' + 
                      sc.data.index[final].strftime('%Y-%m-%d'))
                print(str((sc.data.index[final] - sc.data.index[first]).total_seconds()/3600.) + ' total hours')
                rlonlat = [spice.reclat(np.array(row[['x_pos', 'y_pos', 'z_pos']], dtype='float64')) for indx, row in sc.data.iterrows()]
                rlonlat = np.array(rlonlat).T
                print('Radial range of: ' + str(np.min(rlonlat[0,:])/sc.au_to_km) +
                      ' - ' + str(np.max(rlonlat[0,:])/sc.au_to_km) + ' AU')
                print('Heliolatitude range of: ' + str(np.min(rlonlat[2,:])*180/np.pi) +
                      ' - ' + str(np.max(rlonlat[2,:])*180/np.pi) + 'deg.')
            print('------------------------------------------')
            
            #  If the spacecraft data is split into chunks, we don't want to plot
            #  a continuous line-- so use a collection of patches
            rectangle_list = []
            for x0, xf in zip(start_indx, stop_indx):
                
                ll_corner = (mdates.date2num(sc.data.index[x0]), box_centers[sc.name]-box_height/2.)
                
                width = mdates.date2num(sc.data.index[xf] + dt.timedelta(days=1)) - mdates.date2num(sc.data.index[x0])
                
                rectangle_list.append(patches.Rectangle(ll_corner, width, box_height))
                
                
            #  Unfortunately, there will always be a 1 pixel space between rectangle patches, 
            #  so these can't appear flush against one another when they should...
            axs[0].add_collection(collections.PatchCollection(rectangle_list, facecolors='xkcd:gray', alpha=0.5))
        axs[0].set_xlim((dt.datetime(1970, 1, 1), dt.datetime(1980, 1, 1)))
        
        axs[1].set_ylabel('Observed Radio Flux @ 10.7 cm [SFU]')
        axs[1].set_xlabel('Year')
        #ax0.set_title(r'Spacecraft data availability relative to the Solar Cycle between 4.5-5.5 $R_J$ and -10-10 deg. lat.', wrap=True, fontsize=18)
        
        axs[1].set_ylim((50, 300))
        
        axs[0].set_ylim((0,7))
        axs[0].set_yticks(list(box_centers.values()))
        axs[0].set_yticklabels(list(box_centers.keys()))
        axs[0].set_ylabel('Spacecraft')
        
        #plt.tight_layout()
        figurename = 'Coverage_vs_SolarCycle'
        for suffix in ['.png', '.pdf']:
            plt.savefig('figures/' + figurename + suffix, bbox_inches='tight')
        plt.show()
        spice.kclear()
        
def plot_FullSpacecraftTrajectory(spacecraft_spans):
    import copy
    import spiceypy as spice
    import spacecraftdata as SpacecraftData
    
    # =========================================================================
    #   Would-be inputs
    # =========================================================================
    spacecraft_names = list(spacecraft_spans.keys())
    
    observer1 = 'Solar System Barycenter'
    frame1 = 'JUPITER_JSS'
    observer2 = 'JUPITER BARYCENTER'
    frame2 = 'JUPITER_JSS' 
    
    spice.furnsh('/Users/mrutala/SPICE/customframes/SolarFrames.tf')
    
    #  Get positions for spacecraft relative to observer1/frame1
    spacecraft_list = []
    for name in spacecraft_names:
        sc = SpacecraftData.SpacecraftData(name)
        sc.make_timeseries(*spacecraft_spans[sc.name])
        sc.find_state(frame1, observer1, keep_kernels=True)
        
        spacecraft_list.append(sc)
    spice.furnsh(spacecraft_list[0].SPICE_METAKERNEL_PLANETARY)
    
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        fig, axs = plt.subplots(ncols=2, figsize=(8,4)) # subplot_kw={'projection': 'polar'}
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.05)
            
        for sc in spacecraft_list:
            #  For each spacecraft, get the positions of Earth and Jupiter
            times = [spice.datetime2et(t) for t in sc.data.index]
            
            ert_pos, ear_lt = spice.spkpos('EARTH BARYCENTER', times, frame1, 'NONE', observer1)
            ert_pos /= sc.au_to_km
            
            jup_pos, jup_lt = spice.spkpos('JUPITER BARYCENTER', times, frame1, 'NONE', observer1)
            jup_pos /= sc.au_to_km
            
            xyz_in_AU = np.array([np.array(row[['x_pos', 'y_pos', 'z_pos']]).astype('float64')
                                  for indx, row in sc.data.iterrows()]) / sc.au_to_km 
            print(lat_range)
            sc.find_subset(coord1_range=np.array(r_range)*sc.au_to_km,
                           coord3_range=np.array(lat_range)*np.pi/180., 
                              transform='reclat')
            xyz_sw = sc.data[['x_pos', 'y_pos', 'z_pos']].to_numpy(dtype='float64') / sc.au_to_km
            
            axs[0].plot(ert_pos[:,0], ert_pos[:,1], label='Earth', 
                        color='xkcd:kelly green', marker='o', markersize=0.25, linestyle='None', zorder=-1)
            
            axs[0].plot(jup_pos[:,0], jup_pos[:,1], label='Jupiter', 
                        color='xkcd:peach', marker='o', markersize=0.25, linestyle='None', zorder=-1)
            
            axs[0].plot(xyz_in_AU[:,0], xyz_in_AU[:,1], 
                      label=sc.name, linewidth=1, color='gray', alpha=0.5, zorder=0)
            # axs[0].plot(xyz_sw[:,0], xyz_sw[:,1], 
            #           label=sc.name, linewidth=1, color=pc.spacecraft_colors[sc.name])
            axs[0].scatter(xyz_sw[:,0], xyz_sw[:,1], 
                      label=sc.name, c=pc.spacecraft_colors[sc.name], s=0.5, marker='o', zorder=1)
    
    #  Get positions for spacecraft relative to observer2/frame2
    spacecraft_list = []
    for name in spacecraft_names:
        sc = SpacecraftData.SpacecraftData(name)
        sc.make_timeseries(*spacecraft_spans[sc.name], dt.timedelta(hours=12))
        sc.find_state(frame2, observer2, keep_kernels=True)
        
        spacecraft_list.append(sc)
        
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        for sc in spacecraft_list:
            #  For each spacecraft, get the positions of Earth and Jupiter
            times = [spice.datetime2et(t) for t in sc.data.index]
            
            xyz_in_RJ = np.array([np.array(row[['x_pos', 'y_pos', 'z_pos']]).astype('float64')
                                  for indx, row in sc.data.iterrows()]) / 71492.
            
            #sc.find_subset(coord1_range=np.array(r_range)*sc.au_to_km,
                           #coord3_range=np.array(lat_range)*np.pi/180., 
            #                  transform='reclat')
            sc.find_subset(coord1_range=np.array([120, np.inf])*71492.,
                              transform='reclat')
            
            xyz_sw = sc.data[['x_pos', 'y_pos', 'z_pos']].to_numpy(dtype='float64') / 71492.
            
            axs[1].plot(xyz_in_RJ[:,0], xyz_in_RJ[:,1], 
                      label=sc.name, linewidth=1, color='gray', alpha=0.5, zorder=0)  
            
            axs[1].scatter(xyz_sw[:,0], xyz_sw[:,1], 
                      label=sc.name, linewidth=1, color=pc.spacecraft_colors[sc.name], s=0.5, marker='o', zorder=1)
        
        axs[0].set(xlim=[-6,6], xlabel='JSS X [AU]',
                   ylim=[-6,6], ylabel='JSS Y [AU]',
                   aspect=1)
        axs[0].text(0.05, 0.95, '(a)', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform = axs[0].transAxes)

        
        axs[1].set(xlim=[-600,600], xlabel=r'JSS X [R$_\mathrm{J}$]',
                   ylim=[-600,600], ylabel=r'JSS Y [R$_\mathrm{J}$]',
                   aspect=1)
        axs[1].text(0.05, 0.95, '(b)', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform = axs[1].transAxes)
        
    spice.kclear()
    
    figurename = 'Full_Spacecraft_Trajectory'
    for suffix in ['.png', '.pdf']:
        plt.savefig('figures/' + figurename + suffix, bbox_inches='tight', dpi=300)
    plt.show()
    
    return
    
    
def plot_spacecraft_spatial_coverage():
    import matplotlib.pyplot as plt
    import copy
    import spiceypy as spice
    import spacecraftdata as SpacecraftData
    
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/lsk/latest_leapseconds.tls')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-1.bsp')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-2.bsp')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/pck/pck00011.tpc')
    spice.furnsh('/Users/mrutala/SPICE/customframes/SolarFrames.tf')
    reference_frame = 'SUN_INERTIAL' # HGI  #'ECLIPJ2000'
    observer = 'SUN'

    spacecraft_list = []
    spacecraft_list.append(SpacecraftData.SpacecraftData('Pioneer 10'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Pioneer 11'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Voyager 1'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Voyager 2'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Ulysses'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Juno'))
    
    for sc in spacecraft_list:
        temp_sc = copy.deepcopy(sc)
        temp_sc.find_lifetime()
        temp_sc.make_timeseries()
        temp_sc.find_state(reference_frame, observer)
        temp_sc.find_subset(coord1_range=np.array((4.0, 6.4))*sc.au_to_km, # 4.4 - 6
                            coord3_range=np.array((-8.5,8.5))*np.pi/180., 
                            transform='reclat')
        startdate = temp_sc.data.index[0]
        stopdate = temp_sc.data.index[-1]
        sc.read_processeddata(startdate, stopdate)
        sc.find_state(reference_frame, observer)
        sc.find_subset(coord1_range=np.array((4.0, 6.4))*sc.au_to_km, # 4.4 - 6
                       coord3_range=np.array((-8.5,8.5))*np.pi/180., 
                       transform='reclat')
        
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
    #fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    #plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    #plt.grid(False)
    #plt.xlabel("Ecliptic X")
    #plt.ylabel("Ecliptic Y")
    for ax, sc in zip(axs.reshape(-1),spacecraft_list):
        
        observer='Solar System Barycenter'
        reference_frame = 'ECLIPJ2000'
        sc.find_state(reference_frame, observer, keep_kernels=True)
        
        sc_full = SpacecraftData.SpacecraftData(sc.name)
        sc_full.find_lifetime(keep_kernels=True)
        sc_full.make_timeseries(timedelta=dt.timedelta(days=10))
        sc_full.find_state(reference_frame, observer, keep_kernels=True)
        
        xyz_in_AU = np.array([row[['x_pos', 'y_pos', 'z_pos']]/sc.au_to_km
                              for indx, row in sc.data.iterrows()])
        
        xyz_full_in_AU = np.array([row[['x_pos', 'y_pos', 'z_pos']]/sc.au_to_km
                              for indx, row in sc_full.data.iterrows()])
        
        ax.plot(xyz_full_in_AU[:,0], xyz_full_in_AU[:,1],
                color='gray', linewidth=1)
        ax.plot(xyz_in_AU[:,0], xyz_in_AU[:,1], 
                  label=sc.name, marker='o', markersize=1, linestyle='None', color=spacecraft_colors[sc.name])
        
        ax.set_aspect(1)
    
        times = [spice.str2et(t.strftime('%b %d, %Y %H:%M:%S.%f')) for t in sc.data.index]
        ear_pos, ear_lt = spice.spkpos('EARTH BARYCENTER', times, reference_frame, 'NONE', observer)
        ear_pos = np.array(ear_pos) / sc.au_to_km
        jup_pos, jup_lt = spice.spkpos('JUPITER BARYCENTER', times, reference_frame, 'NONE', observer)
        jup_pos = np.array(jup_pos) / sc.au_to_km
        
        ax.plot(ear_pos[:,0], ear_pos[:,1], 
                label='Earth', color='xkcd:kelly green', marker='o', markersize=1, linestyle='None')
        ax.plot(ear_pos[-1,0], ear_pos[-1,1],  
                color='xkcd:kelly green', marker='o', markersize=4)
        
        ax.plot(jup_pos[:,0], jup_pos[:,1], 
                label='Jupiter', color='xkcd:peach', marker='o', markersize=1, linestyle='None')
        ax.plot(jup_pos[-1,0], jup_pos[-1,1],
                color='xkcd:peach', marker='o', markersize=4)
        
        ax.set(xlim=[-6,6], 
                ylim=[-6,6])
    


