#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:08:47 2023

@author: mrutala

"""
import datetime as dt
import numpy as np
import pandas as pd
import copy 

spacecraft_colors = {'Pioneer 10': '#F6E680',
                     'Pioneer 11': '#FFD485',
                     'Voyager 1' : '#FEC286',
                     'Voyager 2' : '#FEAC86',
                     'Ulysses'   : '#E6877E',
                     'Juno'      : '#D8696D'}
model_colors = {'Tao'    : '#C59FE5',
                'HUXt'   : '#A0A5E4',
                'SWMF-OH': '#98DBEB',
                'MSWIM2D': '#A9DBCA',
                'ENLIL'  : '#DCED96'}
model_symbols = {'Tao'    : 'v',
                'HUXt'   : '*',
                'SWMF-OH': '^',
                'MSWIM2D': 'd',
                'ENLIL'  : 'h'}

r_range = np.array((4.9, 5.5))  #  [4.0, 6.4]
lat_range = np.array((-6.1, 6.1))  #  [-10, 10]


def view_DTWStepPatterns():
    
    
    
    return

def test_PaddedDTW():
    """
    I want to test how DTW algorithms match features from the models to the data
    when the arrays are the same size (and hence same temporal resolution)
    and the start and end are closed (i.e., the first point in both model and data, 
    and the last point in both, must match to each other), 
    BUT with the solar wind data offset by +/- XXX days, to account for overall lags
    
    The idea is to see which model solar wind features most commonly match data solar wind 
    features at a variety of input alignments, and use those statistics to assign matches

    Returns
    -------
    None.

    """
    import matplotlib.pyplot as plt
    import dtw as dtw
    import spiceypy as spice

    import read_SWModel
    import spacecraftdata
    from SWData_Analysis import find_Jumps
    import scipy
    
    # =============================================================================
    #    Would-be inputs go here 
    # =============================================================================
    tag = 'u_mag'  #  solar wind property of interest
    spacecraft_name = 'Juno' # ['Ulysses', 'Juno'] # 'Juno'
    model_name = 'Tao'
    starttime = dt.datetime(1990, 1, 1)
    stoptime = dt.datetime(2017, 1, 1)
    reference_frame = 'SUN_INERTIAL'
    observer = 'SUN'
    intrinsic_error = 1.0 #  This is the error in one direction, i.e. the total width of the arrival time is 2*intrinsic_error
    
    # =============================================================================
    #  Would-be keywords go here
    # =============================================================================
    window_in_days = 13.5
    stepsize_in_days = 0.5
    #resolution_in_min = 60.
    max_lag_in_days = 6.0
    min_lag_in_days = -6.0
    
    spacecraft_temporalresolution_inh = 1.0
    
    #  Parse inputs and keywords
    resolution_str = '{}Min'.format(int(spacecraft_temporalresolution_inh*60.))
    
    # =============================================================================
    #  Read in data; for this, we expect 1 spacecraft and 1 model
    # =============================================================================
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/lsk/latest_leapseconds.tls')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-1.bsp')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-2.bsp')
    spice.furnsh('/Users/mrutala/SPICE/generic/kernels/pck/pck00011.tpc')
    spice.furnsh('/Users/mrutala/SPICE/customframes/SolarFrames.tf')
    
    sc_info = spacecraftdata.SpacecraftData(spacecraft_name)
    sc_info.read_processeddata(starttime, stoptime, resolution=resolution_str, combined=True)
    sc_info.find_state(reference_frame, observer, keep_kernels=True)
    sc_info.find_subset(coord1_range=r_range*sc_info.au_to_km,
                        coord3_range=lat_range*np.pi/180.,
                        transform='reclat')
        
    spice.furnsh(sc_info.SPICE_METAKERNEL)
    ang_datetimes = sc_info.data.index  
    #np.arange(starttime, stoptime, dt.timedelta(days=1)).astype(dt.datetime)
    ets = spice.datetime2et(ang_datetimes)
    ang = [np.abs(spice.trgsep(et, sc_info.name, 'POINT', None, 'Earth', 'POINT', None, 'Sun', 'None')*180/np.pi) for et in ets]
    sc_info.data['a_tse'] = ang
    
    spice.kclear()
    
    model_info = read_SWModel.choose(model_name, spacecraft_name, starttime, stoptime)
      
    spacecraft_data = sc_info.data   #  Template
    model_output = model_info        #  Query
    
    
    # =============================================================================
    #     Find jumps in bulk velocity
    # =============================================================================
    sigma_cutoff = 4
    spacecraft_data = find_Jumps(spacecraft_data, tag, sigma_cutoff, 2.0, resolution_width=0.0)
    model_output = find_Jumps(model_output, tag, sigma_cutoff, 2.0, resolution_width=0.0)
    
    # =============================================================================
    #     Shifting and reindexing the model
    # =============================================================================

    shift_ind, dshift_inh = 3.0, 1.0
    shifts_inh = np.arange(-shift_ind*24, shift_ind*24+dshift_inh, dshift_inh)
    
    out_df = pd.DataFrame(columns=['shift', 'distance', 'normalizeddistance', 'r', 'stddev'])
    for shift in shifts_inh:
        
        model_dt = np.arange(spacecraft_data.index[0]+dt.timedelta(hours=shift), 
                             spacecraft_data.index[-1]+dt.timedelta(hours=shift), 
                             dt.timedelta(hours=spacecraft_temporalresolution_inh)).astype(dt.datetime)
        
        shift_model_output = model_output.reindex(pd.DataFrame(index=model_dt).index, method='nearest')
    
        reference = spacecraft_data['jumps'].to_numpy('float64')
        query = shift_model_output['jumps'].to_numpy('float64')    
        
        window_args = {'window_size': shift_ind*24/spacecraft_temporalresolution_inh}
        # alignment = dtw.dtw(query, reference, keep_internals=True, open_begin=False, open_end=False, 
        #                     step_pattern='symmetricP05', window_type='slantedband', window_args=window_args)
        alignment = dtw.dtw(query, reference, keep_internals=True, 
                            step_pattern='symmetric1',
                            window_type='slantedband', window_args=window_args)
        
        # ## Display the warping curve, i.e. the alignment curve
        # ax = alignment.plot(type="threeway")
        # ax.plot([0, len(query)], [0, len(reference)])
        # ax.text(0.1, 0.9, 'Dist. = ' + '{:8.5}'.format(alignment.distance), transform=ax.transAxes)
        # plt.show()
        
        # fig, ax = plt.subplots()
        # ax.plot(reference, color='black')
        # ax.plot(query + 2)
        # for q_i in np.where(query != 0)[0]:
        #     warped_index = int(np.mean(np.where(alignment.index1 == q_i)[0]))
        #     xcoord = [alignment.index1[warped_index], alignment.index2[warped_index]]
        #     ycoord = [query[xcoord[0]]+2, reference[xcoord[1]]]
        #     ax.plot(xcoord, ycoord, color='xkcd:gray', linestyle=':')
        # plt.show()
        
        #ax = alignment.plot(type='density')
        #plt.show()
        
        wt = dtw.warp(alignment,index_reference=False)
        wt = np.append(wt, len(query)-1)
    
        # fig, ax = plt.subplots()
        # #ax.plot(query[wt])                       
        # #ax.plot(reference, color='black', linestyle=':')
        # # ax.plot(spacecraft_data.index, query[wt])                       
        # # ax.plot(spacecraft_data.index, reference, color='black', linestyle=':')
        # ax.scatter(spacecraft_data.index, shift_model_output['p_dyn'][wt], marker='o')                       
        # ax.plot(spacecraft_data.index, spacecraft_data['p_dyn'], color='black')
        # ax.set_yscale('log')
        # plt.show()
        
        t1 = np.log10(shift_model_output['p_dyn'].to_numpy('float64')[wt])
        t2 = np.log10(spacecraft_data['p_dyn'].to_numpy('float64'))
        
        r = scipy.stats.pearsonr(t1, t2)[0]
        sig = np.std(np.log10(shift_model_output['p_dyn'][wt]))
        
        if (r > out_df['r']).any():
            out_alignment = copy.deepcopy(alignment)
        
        d = {'shift': [shift],
             'distance': [alignment.distance],
             'normalizeddistance': [alignment.normalizedDistance],
             'r': [r],
             'stddev': [sig]}
        out_df = pd.concat([out_df, pd.DataFrame.from_dict(d)], ignore_index=True)
     
    ## Display the warping curve, i.e. the alignment curve
    ax = out_alignment.plot(type="threeway")
    ax.plot([0, len(query)], [0, len(reference)])
    ax.text(0.1, 0.9, 'Dist. = ' + '{:8.5}'.format(out_alignment.distance), transform=ax.transAxes)
    plt.show()
     
    return(out_df, out_alignment)
