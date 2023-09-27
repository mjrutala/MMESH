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
import dtw
import matplotlib.pyplot as plt

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

# def test_PaddedDTW():
#     """
#     I want to test how DTW algorithms match features from the models to the data
#     when the arrays are the same size (and hence same temporal resolution)
#     and the start and end are closed (i.e., the first point in both model and data, 
#     and the last point in both, must match to each other), 
#     BUT with the solar wind data offset by +/- XXX days, to account for overall lags
    
#     The idea is to see which model solar wind features most commonly match data solar wind 
#     features at a variety of input alignments, and use those statistics to assign matches

#     Returns
#     -------
#     None.

#     """
#     import matplotlib.pyplot as plt
#     import dtw as dtw
#     import spiceypy as spice

#     import read_SWModel
#     import spacecraftdata
#     from SWData_Analysis import find_Jumps
#     import scipy
    
#     # =============================================================================
#     #    Would-be inputs go here 
#     # =============================================================================
#     tag = 'u_mag'  #  solar wind property of interest
#     spacecraft_name = 'Juno' # ['Ulysses', 'Juno'] # 'Juno'
#     model_name = 'Tao'
#     starttime = dt.datetime(1990, 1, 1)
#     stoptime = dt.datetime(2017, 1, 1)
#     reference_frame = 'SUN_INERTIAL'
#     observer = 'SUN'
#     intrinsic_error = 1.0 #  This is the error in one direction, i.e. the total width of the arrival time is 2*intrinsic_error
    
#     # =============================================================================
#     #  Would-be keywords go here
#     # =============================================================================
#     window_in_days = 13.5
#     stepsize_in_days = 0.5
#     #resolution_in_min = 60.
#     max_lag_in_days = 6.0
#     min_lag_in_days = -6.0
    
#     spacecraft_temporalresolution_inh = 1.0
    
#     #  Parse inputs and keywords
#     resolution_str = '{}Min'.format(int(spacecraft_temporalresolution_inh*60.))
    
#     # =============================================================================
#     #  Read in data; for this, we expect 1 spacecraft and 1 model
#     # =============================================================================
#     spice.furnsh('/Users/mrutala/SPICE/generic/kernels/lsk/latest_leapseconds.tls')
#     spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-1.bsp')
#     spice.furnsh('/Users/mrutala/SPICE/generic/kernels/spk/planets/de441_part-2.bsp')
#     spice.furnsh('/Users/mrutala/SPICE/generic/kernels/pck/pck00011.tpc')
#     spice.furnsh('/Users/mrutala/SPICE/customframes/SolarFrames.tf')
    
#     sc_info = spacecraftdata.SpacecraftData(spacecraft_name)
#     sc_info.read_processeddata(starttime, stoptime, resolution=resolution_str, combined=True)
#     sc_info.find_state(reference_frame, observer, keep_kernels=True)
#     sc_info.find_subset(coord1_range=r_range*sc_info.au_to_km,
#                         coord3_range=lat_range*np.pi/180.,
#                         transform='reclat')
        
#     spice.furnsh(sc_info.SPICE_METAKERNEL)
#     ang_datetimes = sc_info.data.index  
#     #np.arange(starttime, stoptime, dt.timedelta(days=1)).astype(dt.datetime)
#     ets = spice.datetime2et(ang_datetimes)
#     ang = [np.abs(spice.trgsep(et, sc_info.name, 'POINT', None, 'Earth', 'POINT', None, 'Sun', 'None')*180/np.pi) for et in ets]
#     sc_info.data['a_tse'] = ang
    
#     spice.kclear()
    
#     model_info = read_SWModel.choose(model_name, spacecraft_name, starttime, stoptime)
      
#     spacecraft_data = sc_info.data   #  Template
#     model_output = model_info        #  Query
    
#     #  Get baseline correlation coefficient
#     aligned_model_output = model_output.reindex(spacecraft_data.index, method='nearest')
#     r_baseline = scipy.stats.pearsonr(aligned_model_output['p_dyn'], spacecraft_data['p_dyn'])[0]
    
#     # =============================================================================
#     #     Find jumps in bulk velocity
#     # =============================================================================
#     sigma_cutoff = 4
#     spacecraft_data = find_Jumps(spacecraft_data, tag, sigma_cutoff, 2.0, resolution_width=0.0)
#     model_output = find_Jumps(model_output, tag, sigma_cutoff, 2.0, resolution_width=0.0)
    
#     # =============================================================================
#     #     Shifting and reindexing the model
#     # =============================================================================
#     shift_ind, dshift_inh = 3.0, 1.0
#     shifts_inh = np.arange(-shift_ind*24, shift_ind*24+dshift_inh, dshift_inh)
    
#     out_df = pd.DataFrame(columns=['shift', 'distance', 'normalizeddistance', 'r', 'stddev'])
#     for shift in shifts_inh:
        
#         model_dt = np.arange(spacecraft_data.index[0]+dt.timedelta(hours=shift), 
#                              spacecraft_data.index[-1]+dt.timedelta(hours=shift), 
#                              dt.timedelta(hours=spacecraft_temporalresolution_inh)).astype(dt.datetime)
        
#         shift_model_output = model_output.reindex(pd.DataFrame(index=model_dt).index, method='nearest')
    
#         reference = spacecraft_data['jumps'].to_numpy('float64')
#         query = shift_model_output['jumps'].to_numpy('float64')    
        
#         window_args = {'window_size': shift_ind*24/spacecraft_temporalresolution_inh}
#         # alignment = dtw.dtw(query, reference, keep_internals=True, open_begin=False, open_end=False, 
#         #                     step_pattern='symmetricP05', window_type='slantedband', window_args=window_args)
#         alignment = dtw.dtw(query, reference, keep_internals=True, 
#                             step_pattern='symmetric1',
#                             window_type='slantedband', window_args=window_args)
        
#         # ## Display the warping curve, i.e. the alignment curve
#         # ax = alignment.plot(type="threeway")
#         # ax.plot([0, len(query)], [0, len(reference)])
#         # ax.text(0.1, 0.9, 'Dist. = ' + '{:8.5}'.format(alignment.distance), transform=ax.transAxes)
#         # plt.show()
        
        
#         #ax = alignment.plot(type='density')
#         #plt.show()
        
#         wt = dtw.warp(alignment,index_reference=False)
#         wt = np.append(wt, len(query)-1)
    
#         # fig, ax = plt.subplots()
#         # #ax.plot(query[wt])                       
#         # #ax.plot(reference, color='black', linestyle=':')
#         # # ax.plot(spacecraft_data.index, query[wt])                       
#         # # ax.plot(spacecraft_data.index, reference, color='black', linestyle=':')
#         # ax.scatter(spacecraft_data.index, shift_model_output['p_dyn'][wt], marker='o')                       
#         # ax.plot(spacecraft_data.index, spacecraft_data['p_dyn'], color='black')
#         # ax.set_yscale('log')
#         # plt.show()
        
#         t1 = np.log10(shift_model_output['p_dyn'].to_numpy('float64')[wt])
#         t2 = np.log10(spacecraft_data['p_dyn'].to_numpy('float64'))
        
#         r = scipy.stats.pearsonr(t1, t2)[0]
#         sig = np.std(np.log10(shift_model_output['p_dyn'][wt]))
        
#         if (r > out_df['r']).all():
#             out_alignment = copy.deepcopy(alignment)
#             print(shift)
        
#         d = {'shift': [shift],
#              'distance': [alignment.distance],
#              'normalizeddistance': [alignment.normalizedDistance],
#              'r': [r],
#              'stddev': [sig]}
#         out_df = pd.concat([out_df, pd.DataFrame.from_dict(d)], ignore_index=True)
     
#     ## Display the warping curve, i.e. the alignment curve
#     with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
#         fig, ax = plt.subplots()
#         ax.plot(out_df['shift'], out_df['r'])
#         ax.hlines(y=r_baseline, xmin=out_df['shift'].iloc[0], xmax=out_df['shift'].iloc[-1], color='black', linestyle=':')
#         plt.show()
        
#         ax = out_alignment.plot(type="threeway")
#         ax.plot([0, len(out_alignment.query)], [0, len(out_alignment.reference)])
#         ax.text(0.1, 0.9, 'Dist. = ' + '{:8.5}'.format(out_alignment.distance), transform=ax.transAxes)
    
#         plt.show()
        
#         fig, ax = plt.subplots()
#         ax.plot(out_alignment.reference, color='black')
#         ax.plot(out_alignment.query + 2)
#         for q_i in np.where(out_alignment.query != 0)[0]:
#             warped_index = int(np.mean(np.where(out_alignment.index1 == q_i)[0]))
#             xcoord = [out_alignment.index1[warped_index], out_alignment.index2[warped_index]]
#             ycoord = [out_alignment.query[xcoord[0]]+2, out_alignment.reference[xcoord[1]]]
#             ax.plot(xcoord, ycoord, color='xkcd:gray', linestyle=':')
#         plt.show()
        
#         wt = dtw.warp(out_alignment,index_reference=False)
#         wt = np.append(wt, len(query)-1)
    
#         fig, ax = plt.subplots()
#         #ax.plot(query[wt])                       
#         #ax.plot(reference, color='black', linestyle=':')
#         # ax.plot(spacecraft_data.index, query[wt])                       
#         # ax.plot(spacecraft_data.index, reference, color='black', linestyle=':')
#         ax.scatter(spacecraft_data.index, out_alignment.query[wt], marker='o')                       
#         ax.plot(spacecraft_data.index, out_alignment.reference, color='black')
#         ax.set_yscale('log')
#         plt.show()
     
#     return(out_df, out_alignment)

def find_SolarWindDTW(query_df, reference_df, shift, basis_tag, metric_tag, total_slope_limit=24.0,
                      intermediate_plots=True, model_name=None, spacecraft_name=None):
    import plot_TaylorDiagram as TD
    import scipy 
    window_args = {'window_size': total_slope_limit}
    
    from sklearn.metrics import confusion_matrix
    
    # =============================================================================
    #   Reindex the reference by the shifted amount, 
    #   then reset the index to match the query
    # ============================================================================= 
    # shift_dt = np.arange(query_df.index[0]+dt.timedelta(hours=shift), 
    #                       query_df.index[-1]+dt.timedelta(hours=shift), 
    #                       dt.timedelta(hours=1)).astype(dt.datetime)
    # shift_reference_df = reference_df.reindex(pd.DataFrame(index=shift_dt).index, method='nearest')
    # shift_reference_df.index = query_df.index
    
    # !!!! ADD support for open_end here
    shift_reference_df = reference_df.copy()
    shift_reference_df.index = shift_reference_df.index - dt.timedelta(hours=float(shift))
    shift_reference_df = shift_reference_df.reindex(query_df.index, method='nearest')        
     
    reference = shift_reference_df[basis_tag].to_numpy('float64')
    query = query_df[basis_tag].to_numpy('float64') 
     
    alignment = dtw.dtw(query, reference, keep_internals=True, 
                         step_pattern='symmetric2', open_end=False,
                         window_type='slantedband', window_args=window_args)
     
    tie_points = find_TiePointsInJumps(alignment)
    #tie_points.insert(0, (0,0))  #  As long as open_begin = False
    #tie_points.append((len(query)-1, len(reference)-1))  #  As long as open_end=False
     
    test_arr = np.zeros(len(query))
    for points1, points2 in zip(tie_points, tie_points[1:]):
        #  Get the indices in test_arr that map to equivalent points in the reference
        reference_sample_indx = np.linspace(points1[1], points2[1]-1, points2[0]-points1[0]+1)
        test_arr[points1[0]:points2[0]+1] = reference_sample_indx
     
    interp_ref_basis = np.interp(test_arr, np.arange(0, len(test_arr)), shift_reference_df[basis_tag])
    # test_df = pd.DataFrame(interp_ref_basis,  columns=['ref'], index=query_df.index)
    # test_df = find_Jumps(test_df, 'ref', sigma_cutoff, 2.0, resolution_width=0.0)
    # interp_ref_basis = np.roll(test_df['jumps'].to_numpy('float64'), 1) #  !!!! JUST A BANDAID
    
    #  !!!! May not work with feature widths/implicit errors
    temp_arr = np.zeros(len(query))
    for i, (ref1, ref2) in enumerate(zip(interp_ref_basis[0:-1], interp_ref_basis[1:])):
        if ref1 + ref2 >= 1:
            if ref1 > ref2:
                temp_arr[i] = 1.0
            else:
                temp_arr[i+1] = 1.0
    
    interp_ref_basis = temp_arr
    interp_ref_metric = np.interp(test_arr, np.arange(0,len(test_arr)), shift_reference_df[metric_tag])
    
    cm = confusion_matrix(query, interp_ref_basis, labels=[0,1])
    
    #  The confusion matrix may well have 0s, which is fine
    with np.errstate(divide='log'):
        accuracy = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,0] + cm[0,1] + cm[1,1])
        precision = (cm[1,1])/(cm[0,1] + cm[1,1])
        recall = (cm[1,1])/(cm[1,0] + cm[1,1])
        f1_score = 2 * (precision * recall)/(precision + recall)
    
    t1 = interp_ref_metric
    t2 = query_df[metric_tag].to_numpy('float64')
    
    
    (r, sig), rmse = TD.find_TaylorStatistics(t1, t2)
    #r = scipy.stats.pearsonr(t1, t2)[0]
    #sig = np.std(t1)
    
    if intermediate_plots == True:
        plot_DTWViews(query_df, shift_reference_df, shift, alignment, basis_tag, metric_tag,
                      model_name = model_name, spacecraft_name = spacecraft_name)    
    
    d = {'shift': [shift],
          'distance': [alignment.distance],
          'normalizeddistance': [alignment.normalizedDistance],
          'r': [r],
          'stddev': [sig],
          'rmse': [rmse],
          'true_negative': cm[0,0],
          'true_positive': cm[1,1],
          'false_negative': cm[1,0],
          'false_positive': cm[0,1],
          'accuracy': accuracy,
          'precision': precision,
          'recall': recall,
          'f1_score': f1_score}
    
    zeropoint = shift_reference_df.index[0]
    delta_t = np.interp(test_arr, np.arange(0, len(test_arr)), (shift_reference_df.index - zeropoint).total_seconds()) - (shift_reference_df.index - zeropoint).total_seconds()
    delta_t = pd.DataFrame(data=delta_t, index=shift_reference_df.index, columns=['time_lag'])
    #delta_t = shift_reference_df.index - shift_reference_df.index[0]
    
    return d, delta_t

def find_TiePointsInJumps(alignment, open_end=False, open_begin=False):
    tie_points = []
    for unity in np.where(alignment.query == 1.0)[0]:
        i = np.where(alignment.index1 == unity)[0][0]
        tie_points.append((alignment.index1[i], alignment.index2[i]))
        
    if open_begin == False:
        tie_points.insert(0, (0,0))
    if open_end == False:
        tie_points.append((alignment.N-1, alignment.M-1))
    
    return tie_points

def plot_DTWViews(query_df, reference_df, shift, alignment, basis_tag, metric_tag, 
                  model_name=None, spacecraft_name=None):
    # =============================================================================
    #   Plot:
    #       - Alignment curve 
    #       - Matched features
    #       - Metric time series comparison
    # =============================================================================
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import collections  as mc
    import matplotlib.dates as mdates
    import matplotlib.patheffects as pe
    mosaic =    '''
                BAAAEEE
                BAAAEEE
                BAAAEEE
                .CCCFFF
                .DDDFFF
                .DDDFFF
                '''
                
    model_color = 'red' if model_name == None else model_colors[model_name]
    spacecraft_color = 'blue' if spacecraft_name == None else spacecraft_colors[spacecraft_name]
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
    #with plt.style.context('default'):
        fig, axs = plt.subplot_mosaic(mosaic, figsize=(8,6), 
                                      height_ratios=[1,1,1,1,1,1],
                                      width_ratios=[1,1,1,1,1,1,1])
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.2, wspace=0.1)
        
        #  Approximate 6 major ticks, rounded to occur ever 10n days
        interval = (query_df.index[-1] - query_df.index[0]).days/6.
        interval = 10 * round(interval/10.)
        date_locator = mdates.DayLocator(interval=interval)
        #date_formatter = mdates.DateFormatter('%j')
        
        def date_context_formatter(ticks):
            ticks = [mdates.num2date(t) for t in ticks]

            def date_formatter(tick, pos):
                tick = mdates.num2date(tick)
                tick_str = tick.strftime('%j')
                
                if (pos == 0):
                    tick_str += '\n' + tick.strftime('%d %b')
                    tick_str += '\n' + tick.strftime('%Y')
                 
                try:
                    if (tick.month > ticks[pos-1].month):
                        tick_str += '\n' + tick.strftime('%d %b')
                except:
                    print('FAILED')
                    print(tick.month)
                    print(pos-1)
                    print(ticks[pos-1])
                    
                if (tick.year > ticks[pos-1].year):
                    tick_str += '\n' + tick.strftime('%Y')
                
                return tick_str
            return date_formatter
        
        axs['A'].text(0.01, 1.1, 'Model: ' + model_name, 
                      va='bottom', ha='left', fontsize=10, color=model_color,
                      transform=axs['A'].transAxes)
        axs['A'].text(0.01, 1.02, 'Spacecraft: ' + spacecraft_name, 
                      va='bottom', ha='left', fontsize=10, color=spacecraft_color,
                      transform=axs['A'].transAxes)
        axs['A'].text(0.75, 1.02, 'Input Reference lagged by ' + str(shift) + ' hours',
                      va='bottom', ha='left', fontsize=10,
                      transform=axs['A'].transAxes)
        
        
        axs['A'].plot(query_df.index[alignment.index1], reference_df.index[alignment.index2],
                      color='white', 
                      path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
        axs['A'].plot(axs['A'].get_xlim(), axs['A'].get_xlim(), color='gray', linestyle='--')
        axs['A'].set(xlim=[query_df.index[0], query_df.index[-1]], 
                     ylim=[reference_df.index[0], reference_df.index[-1]])
        
        axs['A'].imshow(alignment.costMatrix, interpolation='None', origin='lower',
                        aspect='auto', 
                        extent=list(axs['A'].get_xlim())+list(axs['A'].get_ylim()))
        
        axs['A'].text(1/3., 2/3., 'reference lags query', 
                      va='center', ha='center', rotation=45, fontsize=10,
                      transform=axs['A'].transAxes)
        axs['A'].text(2/3., 1/3., 'reference leads query', 
                      va='center', ha='center', rotation=45, fontsize=10,
                      transform=axs['A'].transAxes)
        
        axs['A'].text(0.01, 0.99, 'Alignment Curve',
                      va='top', ha='left', fontsize=10,
                      transform=axs['A'].transAxes)
        
        # =====================================================================
        #   Plot the reference and query time series
        # =====================================================================
        axs['B'].plot(reference_df[basis_tag], reference_df.index, color=model_color)
        axs['C'].plot(query_df.index, query_df[basis_tag], color=spacecraft_color)
        
        vertical_shift = np.max(query_df[basis_tag])*1.5
        axs['D'].plot(query_df.index, query_df[basis_tag], color=spacecraft_color)
        axs['D'].plot(reference_df.index, reference_df[basis_tag]+vertical_shift, color=model_color)
        axs['D'].set(ylim=[0,3], yticks=[1, 2.5], yticklabels=['Query', 'Reference'])
        axs['D'].tick_params(axis='y', labelsize=10)
        
        tie_points = find_TiePointsInJumps(alignment)
        connecting_lines = []
        for point in tie_points:
            query_date = mdates.date2num(query_df.index[point[0]])
            reference_date = mdates.date2num(reference_df.index[point[1]])
            connecting_lines.append([(query_date, 1), (reference_date, 2.5)])

        lc = mc.LineCollection(connecting_lines, 
                               linewidths=1, linestyles=":", color='gray', linewidth=2)
        axs['D'].add_collection(lc)
        axs['D'].text(0.01, 0.99, 'Peak Matching',
                      va='top', ha='left', fontsize=10,
                      transform=axs['D'].transAxes)  
        
        test_arr = np.zeros(len(alignment.query))
        for points1, points2 in zip(tie_points, tie_points[1:]):
            #  Get the indices in test_arr that map to equivalent points in the reference
            reference_sample_indx = np.linspace(points1[1], points2[1]-1, points2[0]-points1[0]+1)
            test_arr[points1[0]:points2[0]+1] = reference_sample_indx
        interp_ref_metric = np.interp(test_arr, np.arange(0,len(test_arr)), reference_df[metric_tag])
        
        axs['E'].plot(query_df.index, query_df[metric_tag], color=spacecraft_color)
        axs['E'].plot(reference_df.index, reference_df[metric_tag], color=model_color)
        # axs['E'].text(0.01, 0.90, 'r = ' + f'{r:.2f}', 
        #               va='top', ha='left', fontsize=16,
        #               transform=axs['E'].transAxes)  
        axs['E'].text(0.01, 0.99, 'Reference & Query',
                      va='top', ha='left', fontsize=10,
                      transform=axs['E'].transAxes)  

        axs['F'].plot(query_df.index, query_df[metric_tag], color=spacecraft_color)
        axs['F'].plot(reference_df.index, interp_ref_metric, color=model_color)
        # axs['E'].text(0.01, 0.90, 'r = ' + f'{r:.2f}', 
        #               va='top', ha='left', fontsize=16,
        #               transform=axs['E'].transAxes)  
        axs['F'].text(0.01, 0.99, 'Warped Reference & Query',
                      va='top', ha='left', fontsize=10,
                      transform=axs['F'].transAxes)  
        #print(len(np.where(reference_df[basis_tag] == 1.)[0]))
        #print(np.where(reference_df[basis_tag] == 1.)[0])
        
        for ax in [axs['A'], axs['C'], axs['D'], axs['E'], axs['F']]:
            ax.xaxis.set_major_locator(date_locator)
            ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(date_context_formatter(ax.get_xticks()))
            ax.tick_params(axis='x', labelsize=10)
        for ax in [axs['A'], axs['B']]:
            ax.yaxis.set_major_locator(date_locator)
            ax.yaxis.set_minor_locator(mdates.DayLocator(interval=1))
            ax.yaxis.set_major_formatter(date_context_formatter(ax.get_yticks()))
            ax.tick_params(axis='y', labelsize=10)
        
        axs['A'].set_xticklabels([])
        axs['A'].set_yticklabels([])
        axs['B'].get_xaxis().set_visible(False)
        axs['C'].set_xticklabels([])
        axs['C'].get_yaxis().set_visible(False)
        
        # axs['E'].set(xlim=axs['D'].get_xlim(), xticklabels=[],
        #              yscale='log')
        axs['E'].set_xticklabels([])
        axs['E'].yaxis.tick_right()
        axs['F'].yaxis.tick_right()
    
    plt.show()
    return


def find_OptimalDTW(query_df, reference_df, basis_tag, metric_tag=None, intermediate_plots=None):
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
    
    if metric_tag == None: metric_tag = basis_tag
    total_slope_limit = 4*24.  #  in hours i.e., can't jump more than 4 day
    # =============================================================================
    #     Shifting and reindexing the model
    # =============================================================================    
    start_shift = (reference_df.index[0] - query_df.index[0]).total_seconds()/3600.
    stop_shift = (reference_df.index[-1] - query_df.index[-1]).total_seconds()/3600.
    
    shifts = np.arange(start_shift, stop_shift+1, 1)
    
    out_df = pd.DataFrame(columns=['shift', 'distance', 'normalizeddistance', 
                                   'r', 'stddev', 
                                   'true_negative', 'true_positive', 
                                   'false_negative', 'false_positive'])
    
    for shift in shifts:
        
        d = find_SolarWindDTW(query_df, reference_df, shift, basis_tag, metric_tag, 
                              total_slope_limit = total_slope_limit, intermediate_plots=intermediate_plots)
        out_df = pd.concat([out_df, pd.DataFrame.from_dict(d)], ignore_index=True)
     
        
    return out_df

def compare_SpacecraftAndModels(spacecraft_name, model_names, starttime, stoptime, basis_tag, metric_tag=None):
    

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
    #tag = 'u_mag'  #  solar wind property of interest
    #spacecraft_name = 'Juno'
    #model_name = 'Tao'
    #starttime = dt.datetime(1990, 1, 1)
    #stoptime = dt.datetime(2017, 1, 1)
    reference_frame = 'SUN_INERTIAL'
    observer = 'SUN'
    #intrinsic_error = 1.0 #  This is the error in one direction, i.e. the total width of the arrival time is 2*intrinsic_error
    
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
    spice.furnsh('/Users/mrutala/SPICE/generic/metakernel_planetary.txt')
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
    stats_df = {model_name: pd.DataFrame() for model_name in model_names}
    for model_name in model_names:
        
        model_output = read_SWModel.choose(model_name, spacecraft_name, model_starttime, model_stoptime, resolution='60Min')
        
        # =============================================================================
        #  Get baseline correlation coefficient
        # =============================================================================
        # aligned_model_output = model_output.reindex(spacecraft_data.index, method='nearest')
        # r_baseline = scipy.stats.pearsonr(aligned_model_output['p_dyn'], spacecraft_data['p_dyn'])[0]
        
        # =============================================================================
        #     Find jumps in bulk velocity
        # =============================================================================
        sigma_cutoff = 4
        spacecraft_data = find_Jumps(spacecraft_data, 'u_mag', sigma_cutoff, 2.0, resolution_width=0.0)
        print('Spacecraft done')
        model_output = find_Jumps(model_output, 'u_mag', sigma_cutoff, 2.0, resolution_width=0.0)
        print('Model ' + model_name + ' done')
        
        stats = find_OptimalDTW(spacecraft_data, model_output, basis_tag, metric_tag, intermediate_plots=False)
        stats_df[model_name] = pd.concat([stats_df[model_name], stats], ignore_index=True)
    
    
    
        ref_std = np.nanstd(spacecraft_data[metric_tag])
    return ref_std, stats_df


def plot_TaylorDiagramDTW_FromStats(ref_std, stats_df, metric_tag, filename=''):
    import matplotlib.pyplot as plt
    import plot_TaylorDiagram as TD
    
    # 
    savefile_stem = filename
    if '.' in savefile_stem:
        savefile_stem = savefile_stem.split('.')[0]
        
    match metric_tag:
        case ('u_mag' | 'flow speed'):
            tag = 'u_mag'
            ylabel = r'Solar Wind Flow Speed $u_{mag}$ [km s$^{-1}$]'
            plot_kw = {'yscale': 'linear', 'ylim': (0, 60),
                       'yticks': np.arange(350,550+50,100)}
        case ('p_dyn' | 'pressure'):
            tag = 'p_dyn'
            ylabel = r'Solar Wind Dynamic Pressure $(p_{dyn})$ [nPa]'
            plot_kw = {'yscale': 'log', 'ylim': (0, 0.6),
                       'yticks': 10.**np.arange(-3,0+1,1)}
    
    with plt.style.context('/Users/mrutala/code/python/mjr.mplstyle'):
        
        fig, ax = TD.plot_TaylorDiagram_fromstats(ref_std)
        
        for model_name, stat_df in stats_df.items():
            shifted_stat_plot = ax.scatter(np.arccos(stat_df['r'].to_numpy('float64')), stat_df['stddev'].to_numpy('float64'), 
                                            marker=model_symbols[model_name], s=24, c=stat_df['shift'],
                                            zorder=1)
            
            best_indx = np.argmax(stat_df['r'])
            ax.scatter(np.arccos(stat_df['r'].iloc[best_indx]), stat_df['stddev'].iloc[best_indx], 
                       marker=model_symbols[model_name], s=48, c=model_colors[model_name],
                       edgecolors='black', zorder=2, label=model_name)
            
        ax.set_ylim([0, 1.5*ref_std])
        #ylabel = r'Solar Wind Flow Speed $u_{mag}$ [km s$^{-1}$]'
        ax.text(0.5, 0.9, ylabel,
                horizontalalignment='center', verticalalignment='top',
                transform = ax.transAxes)
        ax.legend(ncols=3, bbox_to_anchor=[0.0,0.0,1.0,0.15], loc='lower left', mode='expand')
        ax.set_axisbelow(True)
        
        plt.colorbar(shifted_stat_plot, location='right', orientation='vertical', 
                      label='Model Shift [hr]',
                      fraction=0.2, pad=0.08, shrink=0.6, aspect=15,
                      ticks=[-72, -48, -24, 0, 24, 48, 72])
        
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    extent = extent.expanded(1.5, 1.0)
    extent.intervalx = extent.intervalx + 0.5

    for suffix in ['.png', '.pdf']:
        plt.savefig('figures/' + savefile_stem, dpi=300, bbox_inches=extent)
    plt.show()            
        