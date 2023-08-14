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
    
    #  Get baseline correlation coefficient
    aligned_model_output = model_output.reindex(spacecraft_data.index, method='nearest')
    r_baseline = scipy.stats.pearsonr(aligned_model_output['p_dyn'], spacecraft_data['p_dyn'])[0]
    
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
        
        if (r > out_df['r']).all():
            out_alignment = copy.deepcopy(alignment)
            print(shift)
        
        d = {'shift': [shift],
             'distance': [alignment.distance],
             'normalizeddistance': [alignment.normalizedDistance],
             'r': [r],
             'stddev': [sig]}
        out_df = pd.concat([out_df, pd.DataFrame.from_dict(d)], ignore_index=True)
     
    ## Display the warping curve, i.e. the alignment curve
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        fig, ax = plt.subplots()
        ax.plot(out_df['shift'], out_df['r'])
        ax.hlines(y=r_baseline, xmin=out_df['shift'].iloc[0], xmax=out_df['shift'].iloc[-1], color='black', linestyle=':')
        plt.show()
        
        ax = out_alignment.plot(type="threeway")
        ax.plot([0, len(out_alignment.query)], [0, len(out_alignment.reference)])
        ax.text(0.1, 0.9, 'Dist. = ' + '{:8.5}'.format(out_alignment.distance), transform=ax.transAxes)
    
        plt.show()
        
        fig, ax = plt.subplots()
        ax.plot(out_alignment.reference, color='black')
        ax.plot(out_alignment.query + 2)
        for q_i in np.where(out_alignment.query != 0)[0]:
            warped_index = int(np.mean(np.where(out_alignment.index1 == q_i)[0]))
            xcoord = [out_alignment.index1[warped_index], out_alignment.index2[warped_index]]
            ycoord = [out_alignment.query[xcoord[0]]+2, out_alignment.reference[xcoord[1]]]
            ax.plot(xcoord, ycoord, color='xkcd:gray', linestyle=':')
        plt.show()
        
        wt = dtw.warp(out_alignment,index_reference=False)
        wt = np.append(wt, len(query)-1)
    
        fig, ax = plt.subplots()
        #ax.plot(query[wt])                       
        #ax.plot(reference, color='black', linestyle=':')
        # ax.plot(spacecraft_data.index, query[wt])                       
        # ax.plot(spacecraft_data.index, reference, color='black', linestyle=':')
        ax.scatter(spacecraft_data.index, out_alignment.query[wt], marker='o')                       
        ax.plot(spacecraft_data.index, out_alignment.reference, color='black')
        ax.set_yscale('log')
        plt.show()
     
    return(out_df, out_alignment)

def find_OptimalDTW(query_df, reference_df, comparison_tag, metric_tag=None):
    """
    Uses open-ended dynamic time warping to find
    Attempts to match the query to the reference for a variety of start times

    Parameters
    ----------
    query_df : Pandas DataFrame
        
    reference_df : Pandas DataFrame
        Must have the same temporal resolution as query_df.
        Must be equal in length or longer than query_df.
        Additional elements will be used to test changing initial offsets.

    Returns
    -------
    None.

    """
    
    if metric_tag == None: metric_tag = comparison_tag
    total_slope_limit = 4*24.  #  in hours i.e., can't jump more than 1 day
    # =============================================================================
    #     Shifting and reindexing the model
    # =============================================================================    
    start_shift = (reference_df.index[0] - query_df.index[0]).total_seconds()/3600.
    stop_shift = (reference_df.index[-1] - query_df.index[-1]).total_seconds()/3600.
    
    shifts = np.arange(start_shift, stop_shift+1, 1)
    print(start_shift, stop_shift)
    out_df = pd.DataFrame(columns=['shift', 'distance', 'normalizeddistance', 'r', 'stddev'])
    for shift in shifts:
        
        # =============================================================================
        #   Reindex the reference by the shifted amount, 
        #   then reset the index to match the query
        # ============================================================================= 
        # shift_dt = np.arange(query_df.index[0]+dt.timedelta(hours=shift), 
        #                      query_df.index[-1]+dt.timedelta(hours=shift), 
        #                      dt.timedelta(hours=1)).astype(dt.datetime)
        # shift_reference_df = reference_df.reindex(pd.DataFrame(index=shift_dt).index, method='nearest')
        # shift_reference_df.index = query_df.index
        
        shift_reference_df = reference_df.copy()
        shift_reference_df.index = shift_reference_df.index - dt.timedelta(hours=shift)
        shift_reference_df = shift_reference_df.reindex(query_df.index, method='nearest')
        print(len(shift_reference_df))
        
    
        reference = shift_reference_df[comparison_tag].to_numpy('float64')
        query = query_df[comparison_tag].to_numpy('float64')    
        
        window_args = {'window_size': total_slope_limit}
        # alignment = dtw.dtw(query, reference, keep_internals=True, open_begin=False, open_end=False, 
        #                     step_pattern='symmetricP05', window_type='slantedband', window_args=window_args)
        alignment = dtw.dtw(query, reference, keep_internals=True, 
                            step_pattern='symmetric2', open_end=True,
                            window_type='slantedband', window_args=window_args)
        
        def plot_DTWViews():
            # =============================================================================
            #   Plot:
            #       - Alignment curve 
            #       - Matched features
            #       - Metric time series comparison
            # =============================================================================
            from matplotlib import collections  as mc
            import matplotlib.dates as mdates
            mosaic =    '''
                        BAAAEEE
                        BAAAEEE
                        BAAADDD
                        .CCCDDD
                        '''
            
            with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
            #with plt.style.context('default'):
                fig, axs = plt.subplot_mosaic(mosaic, figsize=(14,8), 
                                              height_ratios=[2,2,2,1],
                                              width_ratios=[1,2,2,2,2,2,2])
                
                date_locator = mdates.DayLocator(interval=5)
                #date_formatter = mdates.DateFormatter('%j')
                
                def date_context_formatter(ticks):
                    ticks = [mdates.num2date(t) for t in ticks]

                    def date_formatter(tick, pos):
                        tick = mdates.num2date(tick)
                        tick_str = tick.strftime('%j')
                        
                        if (pos == 0):
                            tick_str += '\n' + tick.strftime('%d %b')
                            tick_str += '\n' + tick.strftime('%Y')
                            
                        if (tick.month > ticks[pos-1].month):
                            tick_str += '\n' + tick.strftime('%d %b')
                            
                        if (tick.year > ticks[pos-1].year):
                            tick_str += '\n' + tick.strftime('%Y')
                        
                        return tick_str
                    return date_formatter
                
                axs['B'].plot(shift_reference_df[comparison_tag], shift_reference_df.index, color=model_colors['Tao'])
                axs['C'].plot(query_df.index, query_df[comparison_tag], color=spacecraft_colors['Juno'])
                
                axs['A'].plot(query_df.index[alignment.index1], shift_reference_df.index[alignment.index2])
                axs['A'].plot(axs['A'].get_xlim(), axs['A'].get_xlim(), color='gray', linestyle='--')
                axs['A'].set(xlim=axs['C'].get_xlim(), ylim=axs['B'].get_ylim())
                
                axs['A'].text(1/3., 2/3., 'reference lags query', 
                              va='center', ha='center', rotation=45, fontsize=16,
                              transform=axs['A'].transAxes)
                axs['A'].text(2/3., 1/3., 'reference leads query', 
                              va='center', ha='center', rotation=45, fontsize=16,
                              transform=axs['A'].transAxes)
                
                vertical_shift = np.max(query_df[comparison_tag])*1.5
                axs['D'].plot(query_df.index, query_df[comparison_tag], color=spacecraft_colors['Juno'])
                axs['D'].plot(shift_reference_df.index, shift_reference_df[comparison_tag]+vertical_shift, color=model_colors['Tao'])
                axs['D'].yaxis.tick_right()
                axs['D'].set(yticks=[1, 2.5], yticklabels=['Query', 'Reference'])
                x_tie_points = []
                connecting_lines = []
                for unity in np.where(query == 1.0)[0]:
                    i = np.where(alignment.index1 == unity)[0][0]
                    x_tie_points.append((query_df.index[alignment.index1[i]], 
                                        shift_reference_df.index[alignment.index2[i]]))
                    
                    connecting_lines.append([(mdates.date2num(x_tie_points[-1][0]), 1),
                                             (mdates.date2num(x_tie_points[-1][1]), 2.5)])
                

                lc = mc.LineCollection(connecting_lines, 
                                       linewidths=1, linestyles=":", color='gray', linewidth=2)
                axs['D'].add_collection(lc)
                  
                axs['E'].plot(query_df.index, query_df[metric_tag], color=spacecraft_colors['Juno'])
                axs['E'].plot(shift_reference_df.index, shift_reference_df[metric_tag], color=model_colors['Tao'])

                
                
                for ax in [axs['A'], axs['C'], axs['D'], axs['E']]:
                    ax.xaxis.set_major_locator(date_locator)
                    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
                    ax.xaxis.set_major_formatter(date_context_formatter(ax.get_xticks()))
                for ax in [axs['A'], axs['B']]:
                    ax.yaxis.set_major_locator(date_locator)
                    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
                    ax.yaxis.set_major_formatter(date_context_formatter(ax.get_yticks()))
                
                axs['A'].set_xticklabels([])
                axs['A'].set_yticklabels([])
                
                
                
                axs['E'].set(xlim=axs['D'].get_xlim(), xticklabels=[],
                             yscale='log')
                axs['E'].yaxis.tick_right()
            
            plt.show()
            return
            
        # ## Display the warping curve, i.e. the alignment curve
        # ax = alignment.plot(type="threeway")
        # ax.plot([0, len(query)], [0, len(reference)])
        # ax.text(0.1, 0.9, 'Dist. = ' + '{:8.5}'.format(alignment.distance), transform=ax.transAxes)
        # plt.show()
        
        
        result = plot_DTWViews()
        
        # ax = alignment.plot(type='density')
        # plt.show()
        

    
        # fig, ax = plt.subplots()
        # #ax.plot(query[wt])                       
        # #ax.plot(reference, color='black', linestyle=':')
        # # ax.plot(spacecraft_data.index, query[wt])                       
        # # ax.plot(spacecraft_data.index, reference, color='black', linestyle=':')
        # ax.scatter(shift_reference_df.index[wt], shift_reference_df['p_dyn'][wt], marker='o')                       
        # ax.plot(query_df.index, query_df['p_dyn'], color='black')
        # ax.set_yscale('log')
        # plt.show()
        
        # t1 = np.log10(shift_reference_df['p_dyn'].to_numpy('float64')[wt])
        # t2 = np.log10(query_df['p_dyn'].to_numpy('float64'))
        
        # r = scipy.stats.pearsonr(t1, t2)[0]
        # sig = np.std(np.log10(shift_reference_df['p_dyn'][wt]))
        
        # if (r > out_df['r']).all():
        #     out_alignment = copy.deepcopy(alignment)
        #     print(shift)
        
        # d = {'shift': [shift],
        #      'distance': [alignment.distance],
        #      'normalizeddistance': [alignment.normalizedDistance],
        #      'r': [r],
        #      'stddev': [sig]}
        # out_df = pd.concat([out_df, pd.DataFrame.from_dict(d)], ignore_index=True)
     

    return


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
spacecraft_name = 'Juno'
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
#  Read in spacecraft data; calculate the target-sun-Earth angle
# =============================================================================
sc_info = spacecraftdata.SpacecraftData(spacecraft_name)
sc_info.read_processeddata(starttime, stoptime, resolution=resolution_str, combined=True)
sc_info.find_state(reference_frame, observer)
sc_info.find_subset(coord1_range=r_range*sc_info.au_to_km,
                    coord3_range=lat_range*np.pi/180.,
                    transform='reclat')
    
spice.furnsh(sc_info.SPICE_METAKERNEL)
ang_datetimes = sc_info.data.index  
ets = spice.datetime2et(ang_datetimes)
ang = [np.abs(spice.trgsep(et, sc_info.name, 'POINT', None, 'Earth', 'POINT', None, 'Sun', 'None')*180/np.pi) for et in ets]
sc_info.data['a_tse'] = ang
spice.kclear()

spacecraft_data = sc_info.data
   
# =============================================================================
#  Read in model outputs
# =============================================================================
model_starttime = spacecraft_data.index[0] - dt.timedelta(hours=24*3)
model_stoptime = spacecraft_data.index[-1] + dt.timedelta(hours=24*3+1)  #  Closed interval
model_output = read_SWModel.choose(model_name, spacecraft_name, model_starttime, model_stoptime)

# =============================================================================
#  Get baseline correlation coefficient
# =============================================================================
# aligned_model_output = model_output.reindex(spacecraft_data.index, method='nearest')
# r_baseline = scipy.stats.pearsonr(aligned_model_output['p_dyn'], spacecraft_data['p_dyn'])[0]

# =============================================================================
#     Find jumps in bulk velocity
# =============================================================================
sigma_cutoff = 4
spacecraft_data = find_Jumps(spacecraft_data, tag, sigma_cutoff, 2.0, resolution_width=0.0)
model_output = find_Jumps(model_output, tag, sigma_cutoff, 2.0, resolution_width=0.0)

print(len(spacecraft_data), len(model_output))

test = find_OptimalDTW(spacecraft_data, model_output, 'jumps', 'p_dyn')

    # ## Display the warping curve, i.e. the alignment curve
    # with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
    #     fig, ax = plt.subplots()
    #     ax.plot(out_df['shift'], out_df['r'])
    #     ax.hlines(y=r_baseline, xmin=out_df['shift'].iloc[0], xmax=out_df['shift'].iloc[-1], color='black', linestyle=':')
    #     plt.show()
        
    #     ax = out_alignment.plot(type="threeway")
    #     ax.plot([0, len(out_alignment.query)], [0, len(out_alignment.reference)])
    #     ax.text(0.1, 0.9, 'Dist. = ' + '{:8.5}'.format(out_alignment.distance), transform=ax.transAxes)
    
    #     plt.show()
        
    #     fig, ax = plt.subplots()
    #     ax.plot(out_alignment.reference, color='black')
    #     ax.plot(out_alignment.query + 2)
    #     for q_i in np.where(out_alignment.query != 0)[0]:
    #         warped_index = int(np.mean(np.where(out_alignment.index1 == q_i)[0]))
    #         xcoord = [out_alignment.index1[warped_index], out_alignment.index2[warped_index]]
    #         ycoord = [out_alignment.query[xcoord[0]]+2, out_alignment.reference[xcoord[1]]]
    #         ax.plot(xcoord, ycoord, color='xkcd:gray', linestyle=':')
    #     plt.show()
        
    #     wt = dtw.warp(out_alignment,index_reference=False)
    #     wt = np.append(wt, len(query)-1)
    
    #     fig, ax = plt.subplots()
    #     #ax.plot(query[wt])                       
    #     #ax.plot(reference, color='black', linestyle=':')
    #     # ax.plot(spacecraft_data.index, query[wt])                       
    #     # ax.plot(spacecraft_data.index, reference, color='black', linestyle=':')
    #     ax.scatter(spacecraft_data.index, out_alignment.query[wt], marker='o')                       
    #     ax.plot(spacecraft_data.index, out_alignment.reference, color='black')
    #     ax.set_yscale('log')
    #     plt.show()
     