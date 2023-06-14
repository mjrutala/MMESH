#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:23:16 2023

@author: mrutala

Process in-situ measurements of solar wind dynamic pressure and velocity
in order to return a normalized function showing where jumps in either/both
occur

This is done using a normalized derivative
"""
def SWData_():
    import datetime as dt
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import spiceypy as spice
    import matplotlib.dates as mdates
    
    import sys
    sys.path.append('/Users/mrutala/projects/SolarWindProp/')
    import read_SWData
    
    starttime = dt.datetime(2004, 1, 1)
    finaltime = dt.datetime(2004, 2, 1)
    
    spacecraft_data = read_SWData.Ulysses(starttime, finaltime, basedir='/Users/mrutala/Data/')
    
    
    #  Essential solar system and timekeeping kernels
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/lsk/latest_leapseconds.tls')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-1.bsp')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-2.bsp')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/pck/pck00011.tpc')

    #  Ulysses s/c
    spice.furnsh('/Users/mrutala/SPICE/ulysses/metakernel_ulysses.txt')
    
    sc_et = spice.str2et([d.strftime('%Y-%m-%dT%H:%M:%S.%f') for d in spacecraft_data.index])
    sc_state, sc_lt = spice.spkezr('-55', sc_et, 'ECLIPJ2000', 'NONE', 'SOLAR SYSTEM BARYCENTER')
    sc_state = np.array(sc_state).T
    #sc_state[0:3,*] = 
    
    #  Test smoothing
    halftimedelta = dt.timedelta(hours=2.0)
    
    #  Pandas rolling test
    spacecraft_data['ddt_p_dyn_proton'] = np.gradient(np.log10(spacecraft_data['p_dyn_proton']), 
                                                      spacecraft_data.index.values.astype(np.int64))
    rolling_data = spacecraft_data.rolling(2*halftimedelta, min_periods=1, 
                                          center=True, closed='left')
    
    spacecraft_data['smooth_p_dyn_proton'] = rolling_data['p_dyn_proton'].mean()
    
    spacecraft_data['smooth_ddt_p_dyn_proton'] = np.gradient(np.log10(spacecraft_data['smooth_p_dyn_proton']), 
                                                           spacecraft_data.index.values.astype(np.int64))
    spacecraft_data['norm_smooth_ddt_p_dyn_proton'] = spacecraft_data['smooth_ddt_p_dyn_proton'] / np.std(spacecraft_data['smooth_ddt_p_dyn_proton'])
    
    fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(8,8), height_ratios=[1, 3, 3])
    
    axs[0].plot(spacecraft_data.index, spacecraft_data['rau'], color='black')
    axs[0].set_ylabel('Distance', color='black')
    ax2 = axs[0].twinx()  
    ax2.plot(spacecraft_data.index, spacecraft_data['hlat'], color='red')
    ax2.set_ylabel('Latitude', color='red')
    
    axs[1].plot(spacecraft_data.index, spacecraft_data.p_dyn_proton)
    axs[1].set_yscale('log')
    axs[1].set_ylim(5e-4, 5e0)
    
    axs[2].plot(spacecraft_data.index, spacecraft_data.norm_smooth_ddt_p_dyn_proton)
    axs[2].set_ylim(-10, 10)
    axs[2].axhline(4.5, color='cyan')
    axs[2].axhline(-4.5, color='cyan')
    
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%j"))
    
    plt.show()
    
def plot_spacecraftcoverage_solarcycle():
    import matplotlib.pyplot as plt
    import datetime as dt
    import numpy as np
    
    import pandas as pd
    import spiceypy as spice
    
    #import sys
    #sys.path.append('/Users/mrutala/code/python/')
    import spiceypy_metakernelcontext as spiceypy_mkc
    #sys.path.append('/Users/mrutala/projects/SolarWindEM/')
    #import read_SWData
    import spacecraftdata as SpacecraftData
    
    
    # =============================================================================
    # Find when Ulysses was within +/- 30 deg. of the ecliptic, and near 5 AU
    # =============================================================================
    reference_frame = 'SUN_INERTIAL' # HGI  #'ECLIPJ2000'
    observer = 'SUN'

    #  Essential solar system and timekeeping kernels
    #spice.furnsh('/Users/mrutala/SPICE/generic/kernels/lsk/latest_leapseconds.tls')
    #spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-1.bsp')
    #spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-2.bsp')
    #spice.furnsh('/Users/mrutala/SPICE/generic/kernels/pck/pck00011.tpc')

    #  Ulysses s/c
    #spice.furnsh('/Users/mrutala/SPICE/ulysses/metakernel_ulysses.txt')
    
    #  Custom helio frames
    #spice.furnsh('/Users/mrutala/SPICE/customframes/SolarFrames.tf')
    
    ulysses = SpacecraftData.SpacecraftData('Ulysses')
    ulysses.read_processeddata(starttime, stoptime)
    
    juno = SpacecraftData.SpacecraftData('Juno')
    juno.read_processeddata(starttime, stoptime)
    
    #  It's less efficient, but way easier, to read in all the data and filter later
    sc_daterange = spiceypy_mkc.kernelrange(-55)
    sc_data = read_SWData.Ulysses(sc_daterange[0], sc_daterange[1], basedir='/Users/mrutala/Data/')
    
    sc_et = spice.str2et([d.strftime('%Y-%m-%dT%H:%M:%S.%f') for d in sc_data.index])
    sc_state, sc_lt = spice.spkezr('-55', sc_et, reference_frame, 'NONE', observer)
    
    #sc_state = np.array(sc_state).T
    
    rlonlat = np.array([spice.reclat(state[0:3]) for state in sc_state])
    rlonlat[:,0] /= 1.496e8
    rlonlat[:,1] *= 360./(2*np.pi)
    rlonlat[:,2] *= 360./(2*np.pi)
    
    sc_data['r_eclipj2000'] = rlonlat[:,0]
    sc_data['lon_eclipj2000'] = rlonlat[:,1]
    sc_data['lat_eclipj2000'] = rlonlat[:,2]
    
    sc_state_arr = np.array(sc_state)
    sc_data['x_hgi'] = sc_state_arr[:,0] * 1./1.496e8
    sc_data['y_hgi'] = sc_state_arr[:,1] * 1./1.496e8
    sc_data['z_hgi'] = sc_state_arr[:,2] * 1./1.496e8
    
    #  We want 4.5 =< r =< 5.5 and -30 < lat < 30
    criteria_index = np.where((sc_data['r_eclipj2000'] >= 4.5) 
                            & (sc_data['r_eclipj2000'] <= 5.5)
                            & (sc_data['lat_eclipj2000'] > -10)
                            & (sc_data['lat_eclipj2000'] < 10))
    sc_data = sc_data.iloc[criteria_index]
    
    print(np.shape(criteria_index))
# =============================================================================
#     fig, axs = plt.subplots(nrows=3, sharex=True)
#     axs[0].plot(sc_data.index, sc_data['rau'])
#     axs[0].plot(sc_data.index, rlonlat[:,0])
#     
#     axs[1].plot(sc_data.index, sc_data['hlong'])
#     axs[1].plot(sc_data.index, rlonlat[:,1])
#     
#     axs[2].plot(sc_data.index, sc_data['hlat'])
#     axs[2].plot(sc_data.index, rlonlat[:,2])
# =============================================================================
    
    #axs[0].plot(sc_dates, np.sqrt(sc_state.T[0,:]**2 + sc_state.T[1,:]**2 + sc_state.T[2,:]**2))
    
    #axs[1].plot(sc_dates, sc_state.T[5,:])
    
    # =============================================================================
    # F10.4 Radio flux from the sun
    # =============================================================================
    column_headers = ('date', 'observed_flux', 'adjusted_flux',)
    solar_radio_flux = pd.read_csv('/Users/mrutala/Data/Sun/DRAO/penticton_radio_flux.csv',
                                   header = 0, names = column_headers)
    solar_radio_flux['date'] = [dt.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S') 
                                for d in solar_radio_flux['date']]

    fig, ax0 = plt.subplots(figsize=(16,8))
    ax0.plot(solar_radio_flux['date'], solar_radio_flux['observed_flux'])
    ax0twin = ax0.twinx()
    ax0twin.plot(sc_data.index, np.zeros(len(sc_data.index))+1, linestyle='None', marker='o', color='red')
    
    ax0.set_xlim((dt.datetime(1990, 1, 1), dt.datetime(2010, 1, 1)))
    ax0.set_ylabel('Observed Radio Flux @ 10.7 cm [SFU]', fontsize=18)
    
    ax0.set_title(r'Spacecraft data availability relative to the Solar Cycle between 4.5-5.5 $R_J$ and -10-10 deg. lat.', wrap=True, fontsize=18)
    
    ax0.set_ylim((0, 1000))
    
    spacecraft_labels = {'Ulysses' : 1}
    ax0twin.set_ylim((0,2))
    ax0twin.set_yticks(list(spacecraft_labels.values()))
    ax0twin.set_yticklabels(list(spacecraft_labels.keys()))
    ax0twin.set_ylabel('Spacecraft', fontsize=18)
    ax0twin.tick_params(labelsize=18)
    
    ax0.tick_params(labelsize=18)
    
    plt.show()
    
    return(sc_data) 
    
def SWData_derivativesplots_old():
    
    


    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from astropy.time import Time
    from astropy.io import ascii
    import datetime as dt
    import matplotlib.dates as mdates
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    
    from read_juno_mag_from_amda import juno_b_mag_from_amda
    import read_SWModel as read_SWModel
    import read_SWData as read_SWData
    
    from matplotlib import animation
    
    #timerange = (Time('2001-01-01T00:00:00', format='isot').datetime, Time('2001-03-01T00:00:00', format='isot').datetime)
    target = 'juno'
    timerange = (dt.datetime(2016, 5, 20, 0), dt.datetime(2016, 6, 15, 0))
     
    data_path = ''
    output_figure_name = 'test.png'
    
# =============================================================================
#     Hard-coded results from polling DIAS Magnetospheres group
# =============================================================================
    #  Alexandra, Charlie, Caitriona, Sean, Dale
    results_dict = {'2016-141 12:00:00': 5,
                    '2016-142 12:00:00': 2,
                    '2016-143 03:00:00': 2,
                    '2016-144 18:00:00': 1,
                    '2016-149 12:00:00': 5,
                    '2016-150 03:00:00': 1,
                    '2016-150 18:00:00': 2,
                    '2016-151 12:00:00': 1,
                    '2016-152 12:00:00': 1,
                    '2016-154 21:00:00': 4,
                    '2016-156 00:00:00': 1,
                    '2016-158 06:00:00': 2,
                    '2016-159 21:00:00': 1,
                    '2016-162 12:00:00': 1,
                    '2016-165 18:00:00': 2,
                    }
    results = pd.DataFrame(data=results_dict.values(), 
                           index=[dt.datetime.strptime(t, '%Y-%j %H:%M:%S') for t in results_dict.keys()], 
                           columns=['counts'])


    # =============================================================================
    # Adding Wilson Juno/JADE data
    # =============================================================================
    juno_data = read_SWData.Juno_Wilson2018(timerange[0], timerange[1])
    
    #  Test derivatives
    #d_Umag_dt = np.gradient(juno_data['Umag'], juno_data.index.values.astype(np.int64))
    
    #  Test smoothing
    halftimedelta = dt.timedelta(hours=2.0)
    
    #  This took 18s to run, 100,000x slower than the pandas rolling below
    #timer1 = dt.datetime.now()
    #smooth_Umag = np.empty(len(juno_data))
    #for i, t in enumerate(juno_data.index):
    #    temp = juno_data['Umag'].loc[(juno_data.index >= t - halftimedelta)
    #                                 & (juno_data.index <  t + halftimedelta)]
    #    smooth_Umag[i] = np.mean(temp)
    #print('Time elapsed in smoothing loop: ' + str(dt.datetime.now()-timer1))
    #d_smooth_Umag_dt = np.gradient(smooth_Umag, juno_data.index.values.astype(np.int64))
    #norm_dsU_dt = d_smooth_Umag_dt / np.std(d_smooth_Umag_dt)
    
    #  Pandas rolling test
    juno_data['ddt_pdynproton'] = np.gradient(np.log10(juno_data['pdynproton']), 
                                              juno_data.index.values.astype(np.int64))
    rolling_juno_data = juno_data.rolling(2*halftimedelta, min_periods=1, 
                                          center=True, closed='left')
    juno_data['smooth_pdynproton'] = rolling_juno_data['pdynproton'].mean()
    
    juno_data['smooth_ddt_pdynproton'] = np.gradient(np.log10(juno_data['smooth_pdynproton']), 
                                                     juno_data.index.values.astype(np.int64))
    juno_data['norm_smooth_ddt_pdynproton'] = juno_data['smooth_ddt_pdynproton'] / np.std(juno_data['smooth_ddt_pdynproton'])
    
    #d_smooth_Umag_dt = np.gradient(rolling_juno_data['Umag'].mean(), juno_data.index.values.astype(np.int64))
    #norm_dsU_dt = d_smooth_Umag_dt / np.std(d_smooth_Umag_dt)
    #d_smooth_pdynproton_dt = np.gradient(rolling_juno_data['pdynproton'].mean(), juno_data.index.values.astype(np.int64))
    #norm_dsp_dt = d_smooth_pdynproton_dt / np.std(d_smooth_pdynproton_dt)
    
    # %%===========================================================================
    #  Plot the figure   
    # =============================================================================
    fig = plt.figure(layout='constrained', figsize=(8,11))
    subfigs = fig.subfigures(4, 1, hspace=0.02, height_ratios=[1,1,1,0.2])
    
    axs0 = subfigs[0].subplots(2, 1, sharex=True)
    axs1 = subfigs[1].subplots(2, 1, sharex=True)
    axs2 = subfigs[2].subplots(2, 1, sharex=True)
    cax = subfigs[3].subplots(1, 1)
    
    
    
    # Pressures
    axs0[0].plot(juno_data.index, juno_data.pdynproton, 
                 color='k', linewidth=2, 
                 label='JADE (Wilson+ 2018)')
    axs0[0].fill_between(juno_data.index, 
                         juno_data.pdynproton-juno_data.pdynproton_err, 
                         juno_data.pdynproton+juno_data.pdynproton_err, 
                         step='mid', color='black', alpha=0.5)
    axs0[0].plot(juno_data.index, rolling_juno_data.pdynproton.mean(), 
                 color='orange', linewidth=3,
                 label=r'Boxcar Smoothed $\pm$' + str(halftimedelta))
    
    axs1[0].plot(juno_data.index, juno_data.pdynproton, 'k', label='JADE (Wilson+ 2018)')
    axs1[0].fill_between(juno_data.index, 
                         juno_data.pdynproton-juno_data.pdynproton_err, 
                         juno_data.pdynproton+juno_data.pdynproton_err, 
                         step='mid', color='black', alpha=0.5)
    axs1[0].plot(juno_data.index, rolling_juno_data.pdynproton.mean(), 
                 color='orange', linewidth=3,
                 label=r'Boxcar Smoothed $\pm$' + str(halftimedelta))
    
    axs2[0].plot(juno_data.index, juno_data.pdynproton, 'k', label='JADE (Wilson+ 2018)')
    axs2[0].fill_between(juno_data.index, 
                         juno_data.pdynproton-juno_data.pdynproton_err, 
                         juno_data.pdynproton+juno_data.pdynproton_err, 
                         step='mid', color='black', alpha=0.5)
    axs2[0].plot(juno_data.index, rolling_juno_data.pdynproton.mean(), 
                 color='orange', linewidth=3,
                 label=r'Boxcar Smoothed $\pm$' + str(halftimedelta))

    # Derivatives
    #axs0[1].plot(juno_data.index, juno_data.dpdynproton_dt, 'k', label='JADE (Wilson+ 2018)')
    #axs0[1].fill_between(date_jade_dt, vsw_jade-vsw_err_jade, vsw_jade+vsw_err_jade, step='mid', color='red', alpha=0.5, label=r'$\pm 1 \sigma$')
    axs0[1].plot(juno_data.index, juno_data.norm_smooth_ddt_pdynproton, 'k',
                 label=r'Smoothed + Differentiated + Normalized')
    #axs1[1].plot(juno_data.index, juno_data.dpdynproton_dt, 'k', label='JADE (Wilson+ 2018)')
    #axs1[1].fill_between(date_jade_dt, vsw_jade-vsw_err_jade, vsw_jade+vsw_err_jade, step='mid', color='red', alpha=0.5, label=r'$\pm 1 \sigma$')
    axs1[1].plot(juno_data.index, juno_data.norm_smooth_ddt_pdynproton, 'k',
                 label=r'Smoothed + Differentiated + Normalized')
    #axs2[1].plot(juno_data.index, juno_data.dpdynproton_dt, 'k', label='JADE (Wilson+ 2018)')
    #axs2[1].fill_between(date_jade_dt, vsw_jade-vsw_err_jade, vsw_jade+vsw_err_jade, step='mid', color='red', alpha=0.5, label=r'$\pm 1 \sigma$')
    axs2[1].plot(juno_data.index, juno_data.norm_smooth_ddt_pdynproton, 'k',
                 label=r'Smoothed + Differentiated + Normalized')
    
    #  Assign colors from colormap to selection results
    cmap = plt.get_cmap('plasma')
    bounds = np.arange(0.5, results.counts.max()+1, 1)
    norm = colors.BoundaryNorm(bounds, cmap.N, extend='neither')
    
    for t, r in results.iterrows():
        for axs in (axs0, axs1, axs2,):
            color = cmap((r['counts']-0.5)/results.counts.max())
            alpha = 0.6 # (r['counts']-0.5)/results.counts.max()
            axs[0].axvspan(t-dt.timedelta(hours=4), t+dt.timedelta(hours=4), alpha=alpha, color=color)
            axs[1].axvspan(t-dt.timedelta(hours=4), t+dt.timedelta(hours=4), alpha=alpha, color=color)
    
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='horizontal', label='# of selections', 
             ticks=np.arange(1, results.counts.max()+1, 1))
    
    axs0_range = (dt.datetime(2016, 5, 20, 0), dt.datetime(2016, 6, 2, 0),)
    axs0[1].set_xlim(axs0_range)
    axs0[1].set_xticks(np.arange(axs0_range[0], axs0_range[1], dt.timedelta(days=2)).astype(dt.datetime))
    
    axs1_range = (dt.datetime(2016, 5, 26, 12), dt.datetime(2016, 6, 8, 12),)
    axs1[1].set_xlim(axs1_range)
    test_arr = np.arange(axs1_range[0]+dt.timedelta(hours=12), axs1_range[1]+dt.timedelta(hours=12), dt.timedelta(days=2)).astype(dt.datetime)
    axs1[1].set_xticks(test_arr)
    
    axs2_range = (dt.datetime(2016, 6, 2, 0), dt.datetime(2016, 6, 15, 0))
    axs2[1].set_xlim(axs2_range)   
    axs2[1].set_xticks(np.arange(axs2_range[0], axs2_range[1], dt.timedelta(days=2)).astype(dt.datetime))
    
    for ax in (axs0, axs1, axs2,):
        ax[0].set_ylim(5e-4, 5e0)
        ax[0].set_yscale('log')
        ax[0].set_ylabel(r'$P_{dyn,H^+}$ [nPa]', fontsize=12)
        ax[0].tick_params(labelsize=14)
        
        ax[1].set_ylim(-6, 6)
        ax[1].set_ylabel(r'$\frac{d}{dt}log_{10}(P_{dyn,H^+})$ [$\sigma$]', fontsize=12)
        ax[1].tick_params(labelsize=14)
        #xticks = ax[1].get_xticks()
        #xlabels = ax[1].get_xticklabels()
        #ax[1].set_xticks(xticks)
        #ax[1].set_xticklabels(xlabels, rotation=30, ha='right', rotation_mode='anchor')
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%j"))
        
    axs0[1].set_xlabel(' ', fontsize=14)
    axs1[1].set_xlabel(' ', fontsize=14)
    axs2[1].set_xlabel('Day of Year 2016', fontsize=14)
    plt.suptitle('Cruise solar wind parameters, Juno JADE (Wilson+ 2018) \n ' + 
                 r'Smoothed by $\pm$'+str(halftimedelta), wrap=True, fontsize=14)
    #subfigs[0].align_labels()
    for subfig in subfigs: subfig.align_labels()
    
    plt.savefig('SWFeatureSelection.png', format='png', dpi=500)
    plt.show()
    
# =============================================================================
#     Attempt an animation
# =============================================================================
    #  Initialize figure and axes
    fig = plt.figure(layout='constrained', figsize=(8,11))
    subfigs = fig.subfigures(4, 1, hspace=0.02, height_ratios=[1,1,1,0.2])
    
    axs0 = subfigs[0].subplots(2, 1, sharex=True)
    axs1 = subfigs[1].subplots(2, 1, sharex=True)
    axs2 = subfigs[2].subplots(2, 1, sharex=True)
    cax = subfigs[3].subplots(1, 1)    
    
    #  timedeltas
    halftimedeltas_hours = np.arange(0.05, 8.05, 0.05)
    
    variance = []

    def animate_func(f):
        
        halftimedelta1 = dt.timedelta(hours=halftimedeltas_hours[f])
        
        #  Pandas rolling test
        juno_data['ddt_pdynproton'] = np.gradient(np.log10(juno_data['pdynproton']), 
                                                  juno_data.index.values.astype(np.int64))
        rolling_juno_data = juno_data.rolling(2*halftimedelta1, min_periods=1, 
                                              center=True, closed='left')
        juno_data['smooth_pdynproton'] = rolling_juno_data['pdynproton'].mean()
        
        juno_data['smooth_ddt_pdynproton'] = np.gradient(np.log10(juno_data['smooth_pdynproton']), 
                                                         juno_data.index.values.astype(np.int64))
        juno_data['norm_smooth_ddt_pdynproton'] = juno_data['smooth_ddt_pdynproton'] / np.std(juno_data['smooth_ddt_pdynproton'])
        
        variance.append(np.var(juno_data['smooth_ddt_pdynproton']))
        
        # Clears the figure to update the line, point,   
        for axs in (axs0, axs1, axs2,):
            axs[0].clear()
            axs[1].clear()
        cax.clear()
        
        # Pressures
        axs0[0].plot(juno_data.index, juno_data.pdynproton, 
                     color='k', linewidth=2, 
                     label='JADE (Wilson+ 2018)')
        axs0[0].fill_between(juno_data.index, 
                             juno_data.pdynproton-juno_data.pdynproton_err, 
                             juno_data.pdynproton+juno_data.pdynproton_err, 
                             step='mid', color='black', alpha=0.5)
        axs0[0].plot(juno_data.index, rolling_juno_data.pdynproton.mean(), 
                     color='orange', linewidth=3,
                     label=r'Boxcar Smoothed $\pm$' + str(halftimedelta))
        
        axs1[0].plot(juno_data.index, juno_data.pdynproton, 'k', label='JADE (Wilson+ 2018)')
        axs1[0].fill_between(juno_data.index, 
                             juno_data.pdynproton-juno_data.pdynproton_err, 
                             juno_data.pdynproton+juno_data.pdynproton_err, 
                             step='mid', color='black', alpha=0.5)
        axs1[0].plot(juno_data.index, rolling_juno_data.pdynproton.mean(), 
                     color='orange', linewidth=3,
                     label=r'Boxcar Smoothed $\pm$' + str(halftimedelta))
        
        axs2[0].plot(juno_data.index, juno_data.pdynproton, 'k', label='JADE (Wilson+ 2018)')
        axs2[0].fill_between(juno_data.index, 
                             juno_data.pdynproton-juno_data.pdynproton_err, 
                             juno_data.pdynproton+juno_data.pdynproton_err, 
                             step='mid', color='black', alpha=0.5)
        axs2[0].plot(juno_data.index, rolling_juno_data.pdynproton.mean(), 
                     color='orange', linewidth=3,
                     label=r'Boxcar Smoothed $\pm$' + str(halftimedelta))

        # Derivatives
        #axs0[1].plot(juno_data.index, juno_data.dpdynproton_dt, 'k', label='JADE (Wilson+ 2018)')
        #axs0[1].fill_between(date_jade_dt, vsw_jade-vsw_err_jade, vsw_jade+vsw_err_jade, step='mid', color='red', alpha=0.5, label=r'$\pm 1 \sigma$')
        axs0[1].plot(juno_data.index, juno_data.norm_smooth_ddt_pdynproton, 'k',
                     label=r'Smoothed + Differentiated + Normalized')
        #axs1[1].plot(juno_data.index, juno_data.dpdynproton_dt, 'k', label='JADE (Wilson+ 2018)')
        #axs1[1].fill_between(date_jade_dt, vsw_jade-vsw_err_jade, vsw_jade+vsw_err_jade, step='mid', color='red', alpha=0.5, label=r'$\pm 1 \sigma$')
        axs1[1].plot(juno_data.index, juno_data.norm_smooth_ddt_pdynproton, 'k',
                     label=r'Smoothed + Differentiated + Normalized')
        #axs2[1].plot(juno_data.index, juno_data.dpdynproton_dt, 'k', label='JADE (Wilson+ 2018)')
        #axs2[1].fill_between(date_jade_dt, vsw_jade-vsw_err_jade, vsw_jade+vsw_err_jade, step='mid', color='red', alpha=0.5, label=r'$\pm 1 \sigma$')
        axs2[1].plot(juno_data.index, juno_data.norm_smooth_ddt_pdynproton, 'k',
                     label=r'Smoothed + Differentiated + Normalized')
        
        #  Assign colors from colormap to selection results
        cmap = plt.get_cmap('plasma')
        bounds = np.arange(0.5, results.counts.max()+1, 1)
        norm = colors.BoundaryNorm(bounds, cmap.N, extend='neither')
        
        for t, r in results.iterrows():
            for axs in (axs0, axs1, axs2,):
                color = cmap((r['counts']-0.5)/results.counts.max())
                alpha = 0.6 # (r['counts']-0.5)/results.counts.max()
                axs[0].axvspan(t-dt.timedelta(hours=4), t+dt.timedelta(hours=4), alpha=alpha, color=color)
                axs[1].axvspan(t-dt.timedelta(hours=4), t+dt.timedelta(hours=4), alpha=alpha, color=color)
        
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=cax, orientation='horizontal', label='# of selections', 
                 ticks=np.arange(1, results.counts.max()+1, 1))
        
        axs0_range = (dt.datetime(2016, 5, 20, 0), dt.datetime(2016, 6, 2, 0),)
        axs0[1].set_xlim(axs0_range)
        axs0[1].set_xticks(np.arange(axs0_range[0], axs0_range[1], dt.timedelta(days=2)).astype(dt.datetime))
        
        axs1_range = (dt.datetime(2016, 5, 26, 12), dt.datetime(2016, 6, 8, 12),)
        axs1[1].set_xlim(axs1_range)
        test_arr = np.arange(axs1_range[0]+dt.timedelta(hours=12), axs1_range[1]+dt.timedelta(hours=12), dt.timedelta(days=2)).astype(dt.datetime)
        axs1[1].set_xticks(test_arr)
        
        axs2_range = (dt.datetime(2016, 6, 2, 0), dt.datetime(2016, 6, 15, 0))
        axs2[1].set_xlim(axs2_range)   
        axs2[1].set_xticks(np.arange(axs2_range[0], axs2_range[1], dt.timedelta(days=2)).astype(dt.datetime))
        
        for ax in (axs0, axs1, axs2,):
            ax[0].set_ylim(5e-4, 5e0)
            ax[0].set_yscale('log')
            ax[0].set_ylabel(r'$P_{dyn,H^+}$ [nPa]', fontsize=12)
            ax[0].tick_params(labelsize=14)
            
            ax[1].set_ylim(-6, 6)
            ax[1].set_ylabel(r'$\frac{d}{dt}log_{10}(P_{dyn,H^+})$ [$\sigma$]', fontsize=12)
            ax[1].tick_params(labelsize=14)
            #xticks = ax[1].get_xticks()
            #xlabels = ax[1].get_xticklabels()
            #ax[1].set_xticks(xticks)
            #ax[1].set_xticklabels(xlabels, rotation=30, ha='right', rotation_mode='anchor')
            ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%j"))
            
        axs0[1].set_xlabel(' ', fontsize=14)
        axs1[1].set_xlabel(' ', fontsize=14)
        axs2[1].set_xlabel('Day of Year 2016', fontsize=14)
        plt.suptitle('Cruise solar wind parameters, Juno JADE (Wilson+ 2018) \n ' + 
                     r'Smoothed by $\pm$'+str(halftimedelta1), wrap=True, fontsize=14)
        #subfigs[0].align_labels()
        for subfig in subfigs: subfig.align_labels()
        
    anim = animation.FuncAnimation(fig, animate_func, interval=1000,   
                                   frames=len(halftimedeltas_hours))
    FFwriter = animation.FFMpegWriter(fps=6)
    anim.save('DifferingSmoothingWindows.mp4', writer=FFwriter)
    
    fig, ax = plt.subplots()
    
    ax.plot(halftimedeltas_hours, variance[1:])
    ax.set_xlabel('Smoothing half width [hours]')
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_ylabel(r'$\sigma^2$ of the smoothed and differentiated $log_{10}(P_{dyn})$')
    
    plt.savefig('SWFeatureVariance_withsmoothing.png', format='png', dpi=500)
    plt.show()
    
    return(variance)