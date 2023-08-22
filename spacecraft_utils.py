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



def plot_spacecraftcoverage_solarcycle():
    import copy
    import matplotlib.pyplot as plt
    #import datetime as dt
    #import numpy as np
    
    #import pandas as pd
    import spiceypy as spice
    
    #import sys
    #sys.path.append('/Users/mrutala/projects/SolarWindEM/')
    #import read_SWData
    import spacecraftdata as SpacecraftData
    
    r_range = [4.9, 5.5]  #  [4.0, 6.4]
    lat_range = [-6.1, 6.1]  #  [-10, 10]
    
    # =============================================================================
    # Find when Ulysses was within +/- 30 deg. of the ecliptic, and near 5 AU
    # =============================================================================
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
        # temp_sc = copy.deepcopy(sc)
        # temp_sc.find_lifetime()
        # temp_sc.make_timeseries()
        # temp_sc.find_state(reference_frame, observer)
        # temp_sc.find_subset(coord1_range=np.array(r_range)*sc.au_to_km, # 4.4 - 6
        #                     coord3_range=np.array(lat_range)*np.pi/180., 
        #                     transform='reclat')
        # startdate = temp_sc.data.index[0]
        # stopdate = temp_sc.data.index[-1]
        sc.read_processeddata(everything=True)
        sc.find_state(reference_frame, observer)
        sc.find_subset(coord1_range=np.array(r_range)*sc.au_to_km, # 4.4 - 6
                       coord3_range=np.array(lat_range)*np.pi/180., 
                       transform='reclat')
    
    spacecraft_labels = {'Pioneer 10': 1,
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
        
        for sc in spacecraft_list:
            
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
            # rectangle_list = []
            # for x0, xf in zip(np.insert(start_indx, 0, 0), stop_indx):
            #     ll_corner = (sc.data.index[x0].timestamp(), spacecraft_labels[sc.name]-0.333)
            #     width = sc.data.index[xf-1].timestamp() - sc.data.index[x0].timestamp()
            #     height = 0.666
                
            #     rectangle_list.append(patch.Rectangle(ll_corner, width, height))
            # ax0twin.add_collection(PatchCollection(rectangle_list))
            
            axs[0].plot(sc.data.index, np.zeros(len(sc.data.index))+spacecraft_labels[sc.name], 
                        linestyle='None', marker='|', markersize=12, color=spacecraft_colors[sc.name])
            
        axs[0].set_xlim((dt.datetime(1970, 1, 1), dt.datetime(2020, 1, 1)))
        
        axs[1].set_ylabel('Observed Radio Flux @ 10.7 cm [SFU]')
        axs[1].set_xlabel('Year')
        #ax0.set_title(r'Spacecraft data availability relative to the Solar Cycle between 4.5-5.5 $R_J$ and -10-10 deg. lat.', wrap=True, fontsize=18)
        
        axs[1].set_ylim((50, 300))
        
        axs[0].set_ylim((0,7))
        axs[0].set_yticks(list(spacecraft_labels.values()))
        axs[0].set_yticklabels(list(spacecraft_labels.keys()))
        axs[0].set_ylabel('Spacecraft')
        
        #plt.tight_layout()
        figurename = 'Coverage_vs_SolarCycle'
        for suffix in ['.png', '.pdf']:
            plt.savefig('figures/' + figurename + suffix, bbox_inches='tight')
        plt.show()
        spice.kclear()
        
def plot_SpacecraftSpatialCoverage():
    import matplotlib.pyplot as plt
    import copy
    import spiceypy as spice
    import spacecraftdata as SpacecraftData
    
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/lsk/latest_leapseconds.tls')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-1.bsp')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-2.bsp')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/pck/pck00011.tpc')
    spice.furnsh('/Users/mrutala/SPICE/customframes/SolarFrames.tf')
    observer='Solar System Barycenter'
    reference_frame = 'ECLIPJ2000'

    spacecraft_list = []
    spacecraft_list.append(SpacecraftData.SpacecraftData('Pioneer 10'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Pioneer 11'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Voyager 1'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Voyager 2'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Ulysses'))
    spacecraft_list.append(SpacecraftData.SpacecraftData('Juno'))
    
    for sc in spacecraft_list:
        #sc.find_lifetime()
        #sc.make_timeseries()
        sc.read_processeddata(everything=True)
        sc.find_state(reference_frame, observer, keep_kernels=True)
        
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
        

    
    for sc in spacecraft_list:
        #  For each spacecraft, get the positions of Earth and Jupiter
        times = [spice.datetime2et(t) for t in sc.data.index]
        print(sc.name)
        print(sc.data.index[0])
        print(sc.data.index[-1])
        
        ear_pos, ear_lt = spice.spkpos('EARTH BARYCENTER', times, reference_frame, 'NONE', observer)
        ear_pos = np.array([spice.recsph(row / sc.au_to_km) for row in ear_pos])
        
        jup_pos, jup_lt = spice.spkpos('JUPITER BARYCENTER', times, reference_frame, 'NONE', observer)
        jup_pos = np.array([spice.recsph(row / sc.au_to_km) for row in jup_pos])
        
        xyz_in_AU = [np.array(row[['x_pos', 'y_pos', 'z_pos']]).astype('float64')
                              for indx, row in sc.data.iterrows()]
        sph_in_AU = np.array([spice.recsph(row / sc.au_to_km) for row in xyz_in_AU])
        #print(sph_in_AU)
        #xyz_full_in_AU = np.array([row[['x_pos', 'y_pos', 'z_pos']]/sc.au_to_km
        #                      for indx, row in sc_full.data.iterrows()])
        
        #ax.plot(xyz_full_in_AU[:,0], xyz_full_in_AU[:,1],
        #        color='gray', linewidth=1)
        ax.plot((sph_in_AU[:,2]-jup_pos[:,2]), sph_in_AU[:,0], 
                  label=sc.name, linewidth=1, color=spacecraft_colors[sc.name])
        
        ax.plot((ear_pos[:,2]-jup_pos[:,2]), ear_pos[:,0], 
                label='Earth', color='xkcd:kelly green', marker='o', markersize=1, linestyle='None')
        #ax.plot(ear_pos[-1,0], ear_pos[-1,1],  
        #        color='xkcd:kelly green', marker='o', markersize=4)
        
        ax.plot((jup_pos[:,2]-jup_pos[:,2]), jup_pos[:,0], 
                label='Jupiter', color='xkcd:peach', marker='o', markersize=1, linestyle='None')
        
    
    ax.set_aspect(1)
    ax.set_rlim((0, 6))

    return(jup_pos)
    
    
    #ax.plot(jup_pos[-1,0], jup_pos[-1,1],
    #        color='xkcd:peach', marker='o', markersize=4)
    
    #ax.set(xlim=[-6,6], 
    #        ylim=[-6,6])
    
    plt.show()
    spice.kclear()
    
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
    


