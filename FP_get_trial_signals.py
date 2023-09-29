from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from copy import deepcopy
import numpy as np
from scipy.integrate import simps
from re import split, search
from glob import glob
from os.path import sep
from os import remove, makedirs
import csv
import matplotlib.pyplot as plt
from matplotlib import patches
import json
import platform
from datetime import datetime
import warnings
from multiprocessing import Pool, cpu_count, freeze_support
from format_axes import format_ax

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


#
# # Set plotting parameters
# label_font_size = 11
# tick_label_size = 7
# legend_font_size = 6
# line_thickness = 1
#
# rcParams['figure.dpi'] = 600
# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42
# rcParams['font.family'] = 'Arial'
# rcParams['font.weight'] = 'bold'
# rcParams['axes.labelweight'] = 'bold'
#
# rcParams['font.size'] = label_font_size
# rcParams['axes.labelsize'] = label_font_size
# rcParams['axes.titlesize'] = label_font_size
# rcParams['axes.linewidth'] = line_thickness
# rcParams['legend.fontsize'] = legend_font_size
# rcParams['xtick.labelsize'] = tick_label_size
# rcParams['ytick.labelsize'] = tick_label_size
# rcParams['errorbar.capsize'] = label_font_size
# rcParams['lines.markersize'] = line_thickness
# rcParams['lines.linewidth'] = line_thickness


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def calculate_PeakValue_and_AUC(sigs, trial_info, baseline_window_start_time, trial_type,
                                fs, x_axis,
                                fixed_trapz_start=0, fixed_trapz_duration=4, fixed_auc_window=True,
                                spout_key_times=None):
    auc_response = np.zeros(np.shape(sigs)[0])
    peak = np.zeros(np.shape(sigs)[0])
    auc_baseline = np.zeros(np.shape(sigs)[0])
    ret_trial_info = deepcopy(trial_info)  # Make a copy in case this needs to be modified
    for trial_idx, cur_trial in enumerate(ret_trial_info):
        if not fixed_auc_window and spout_key_times is not None:
            # Find spout offset that triggered Hit/FA or spout offset around a shock
            if trial_type == 'Hit' or trial_type == 'False alarm':
                cur_trial_onset = cur_trial[2]
                cur_trial_offset = cur_trial[3]
                spoutOffset_triggers = spout_key_times[(spout_key_times['Spout_offset'] > cur_trial_onset) &
                                                       (spout_key_times['Spout_offset'] < cur_trial_offset)][
                    'Spout_offset'].values
                try:
                    trapz_start = spoutOffset_triggers[-1] - cur_trial_onset

                except IndexError:
                    # passive recordings or something weird; could also be the last trial, in which case,
                    # skip measurement and remove trial from trial_info

                    # trapz_start = fixed_trapz_start
                    del ret_trial_info[trial_idx]
                    continue


            # Find spout offset occurring right after a shock, say 0.5 s after shock duration?
            # Plot spout offset latency after shock onset to get an idea
            elif trial_type == 'Miss (shock)':
                shock_duration = 0.3  # seconds
                response_window = 1  # seconds
                cur_trial_onset = cur_trial[2]
                cur_trial_offset = cur_trial[3]
                spoutOffset_triggers = spout_key_times[
                    (spout_key_times['Spout_offset'] > cur_trial_offset) &
                    (spout_key_times['Spout_offset'] < cur_trial_offset + response_window)]['Spout_offset'].values
                try:
                    trapz_start = spoutOffset_triggers[0] - cur_trial_onset

                except IndexError:
                    # passive recordings or something weird; could also be the last trial, in which case,
                    # skip measurement and remove trial from trial_info

                    # trapz_start = fixed_trapz_start
                    del ret_trial_info[trial_idx]
                    continue
            else:
                trapz_start = fixed_trapz_start

        else:
            trapz_start = fixed_trapz_start

        trapz_end = trapz_start + fixed_trapz_duration
        bounded_response_xaxis = x_axis[int((trapz_start + baseline_window_start_time) * fs):
                                        int((trapz_end + baseline_window_start_time) * fs)]

        bounded_response = sigs[trial_idx, int((trapz_start + baseline_window_start_time) * fs):
                                           int((trapz_end + baseline_window_start_time) * fs)]

        bounded_baseline_xaxis = x_axis[0:int(baseline_window_start_time * fs)]
        bounded_baseline = sigs[trial_idx, 0:int(baseline_window_start_time * fs)]

        auc_response[trial_idx] = simps(bounded_response, bounded_response_xaxis)
        peak[trial_idx] = np.max(bounded_response)
        auc_baseline[trial_idx] = simps(bounded_baseline, bounded_baseline_xaxis)

    return ret_trial_info, auc_response, peak, auc_baseline


def run_pipeline(input_list):
    (baseline_window_start_time, response_window_duration, keys_path, output_path, fixed_auc_window), (
        memory_path, all_json) = input_list
    # Split path name to get subject, session and unit ID for prettier output
    split_memory_path = split(REGEX_SEP, memory_path)  # split path
    recording_id = split_memory_path[-1][:-4]  # Example id: SUBJ-ID-104_FP-Aversive-AM-210707-110339_dff

    split_timestamps_name = split("_*_", recording_id)[1]  # split timestamps
    cur_date = split("-*-", split_timestamps_name)[3]
    cur_timestamp = split("-*-", split_timestamps_name)[4]

    subject_id = split("_*_", recording_id)[0]

    # For debugging purposes
    # if unit_id != "SUBJ-ID-197_210511_concat_cluster820":
    #     continue
    # first_entry_flag = True

    # Use subj-session identifier to grab appropriate key
    # Stimulus info is in trialInfo

    # These are in alphabetical order. Must sort by date_trial or match with filev
    # Match by name for now for breakpoints

    key_paths_info = glob(keys_path + sep + subject_id + '*' + 'Aversive*' +
                          cur_date + "*_trialInfo.csv")
    key_paths_spout = glob(keys_path + sep + subject_id + '*' + 'Aversive*' +
                           cur_date + "*spoutTimestamps.csv")

    if len(key_paths_info) == 0:
        print("Key not found for " + recording_id)
        return

    # Load key and spout files
    info_key_times = pd.read_csv(key_paths_info[0])
    spout_key_times = pd.read_csv(key_paths_spout[0])

    # Load signal
    processed_signal = pd.read_csv(memory_path)

    fs = 1 / np.mean(np.diff(processed_signal['Time']))

    # Grab GO trials and FA trials for stim response then walk back to get the immediately preceding CR trial
    # One complicating factor is spout onset/offset but I'll ignore this for now
    # Also ignore reminder trials

    hit_key_times = info_key_times[
        (info_key_times['TrialType'] == 0) & (info_key_times['Hit'] == 1) & (info_key_times['Reminder'] == 0)]
    miss_key_times = info_key_times[
        (info_key_times['TrialType'] == 0) & (info_key_times['Miss'] == 1) & (info_key_times['Reminder'] == 0)]
    fa_key_times = info_key_times[
        (info_key_times['FA'] == 1) & (info_key_times['Reminder'] == 0)]

    # Keep track of trial number and onset time too
    def __get_trialID_zscore(key_times_df):
        baseline_duration = 1  # in seconds; from baseline_start_time
        ret_list = list()
        for _, cur_trial in key_times_df.iterrows():
            signal_around_trial = processed_signal[
                (processed_signal['Time'] >= (cur_trial['Trial_onset'] - baseline_window_start_time)) &
                (processed_signal['Time'] < (cur_trial['Trial_onset'] + response_window_duration))]
            # z-score it

            # 405 fit-removed signal
            baseline_signal = processed_signal[
                (processed_signal['Time'] >= (cur_trial['Trial_onset'] - baseline_window_start_time)) &
                (processed_signal['Time'] < (
                        cur_trial['Trial_onset'] - baseline_window_start_time + baseline_duration))]['Ch465_dff']

            # Non-corrected 465 signal
            # baseline_signal = processed_signal[
            #     (processed_signal['Time'] >= (cur_trial['Trial_onset'] - BASELINE_DURATION_FOR_ZSCORE)) &
            #     (processed_signal['Time'] < (cur_trial['Trial_onset'] - BASELINE_DURATION_FOR_ZSCORE + 1))]['Ch465_mV']

            baseline_mean = np.nanmean(baseline_signal)
            baseline_std = np.nanstd(baseline_signal)

            # 405 fit-removed signal
            dff_zscore = (signal_around_trial['Ch465_dff'].values - baseline_mean) / baseline_std

            # Non-corrected 465 signal
            # dff_zscore = (signal_around_trial['Ch465_mV'].values - baseline_mean) / baseline_std

            # Trial parameters will be under index 0, signal will always be index 1
            ret_list.append(((cur_trial['TrialID'], cur_trial['AMdepth'],
                              cur_trial['Trial_onset'],
                              cur_trial['Trial_offset']),
                             dff_zscore))
        return ret_list

    hit_signals = __get_trialID_zscore(hit_key_times)
    missShock_signals = __get_trialID_zscore(miss_key_times[miss_key_times['ShockFlag'] == 1])
    missNoShock_signals = __get_trialID_zscore(miss_key_times[miss_key_times['ShockFlag'] == 0])
    fa_signals = __get_trialID_zscore(fa_key_times)

    # uniformize lengths and exclude truncated signals by more than half sampling rate points
    # The median length should be the target
    tolerance = fs / 2
    median_length = np.median([item for sublist in
                               [[len(x[1]) for x in hit_signals], [len(x[1]) for x in missShock_signals],
                                [len(x[1]) for x in fa_signals]] for item in sublist])

    hit_signals = [x for x in hit_signals if (len(x[1]) > median_length - tolerance) and
                   (len(x[1]) < median_length + 100)]
    missShock_signals = [x for x in missShock_signals if (len(x[1]) > median_length - tolerance) and
                         (len(x[1]) < median_length + 100)]
    missNoShock_signals = [x for x in missNoShock_signals if (len(x[1]) > median_length - tolerance) and
                           (len(x[1]) < median_length + 100)]
    fa_signals = [x for x in fa_signals if (len(x[1]) > median_length - tolerance) and
                  (len(x[1]) < median_length + 100)]

    # Now uniformize lengths (tolerated jitter of 1 point)
    min_length = np.min([item for sublist in
                         [[len(x[1]) for x in hit_signals], [len(x[1]) for x in missShock_signals],
                          [len(x[1]) for x in missNoShock_signals],
                          [len(x[1]) for x in fa_signals]] for item in sublist])

    hit_signals = [(x[0], np.array(x[1][0:min_length])) for x in hit_signals]
    missShock_signals = [(x[0], np.array(x[1][0:min_length])) for x in missShock_signals]
    missNoShock_signals = [(x[0], np.array(x[1][0:min_length])) for x in missNoShock_signals]
    fa_signals = [(x[0], np.array(x[1][0:min_length])) for x in fa_signals]

    hit_color = '#60B2E5'
    missShock_color = '#C84630'
    missNoShock_color = '#C84630'
    fa_color = 'goldenrod'

    output_dict = dict()
    peak_list = list()
    trial_type_list = list()
    trapz_start = 0
    # trapz_duration = 4  # used for fixed trapz start
    trapz_duration = 3  # used for trial-by-trial trapz start
    with PdfPages(sep.join([output_path, recording_id + '_trialSignals.pdf'])) as pdf:
        # fig, ax = plt.subplots(1, 1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Trial shading
        ax.axvspan(0, 1, facecolor='black', alpha=0.1)
        legend_handles = list()
        for trial_sig, color, trial_type in \
                zip([hit_signals, missNoShock_signals, missShock_signals, fa_signals],
                    [hit_color, missNoShock_color, missShock_color, fa_color],
                    ['Hit', 'Miss (no shock)', 'Miss (shock)', 'False alarm']):
            # Separate trial info for simplicity in calculations below
            # trialID = np.array([ts[0][0] for ts in trial_sig])
            # AMdepth = np.array([ts[0][1] for ts in trial_sig])
            # trialOnset =np.array([ts[0][2] for ts in trial_sig])

            sigs = np.array([ts[1] for ts in trial_sig], ndmin=2)

            if np.size(sigs) == 0:
                continue

            # trial_type_list.append(trial_type)
            if trial_type == 'Miss (no shock)':
                linestyle = '--'
            else:
                linestyle = '-'
            signals_mean = np.nanmean(sigs, axis=0)
            signals_std = np.nanstd(sigs, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(sigs), axis=0))
            x_axis = np.linspace(-baseline_window_start_time, response_window_duration, len(signals_mean))
            ax.plot(x_axis, signals_mean, color=color, linestyle=linestyle)
            ax.fill_between(x_axis, signals_mean - signals_std, signals_mean + signals_std,
                            alpha=0.1, color=color, edgecolor='none')

            legend_handles.append(patches.Patch(facecolor=color, edgecolor=None, alpha=0.5,
                                                label=trial_type))

            # Measure and add measurements to list
            trial_info = [x[0] for x in trial_sig]  # idx=0
            trial_info, auc_response, peak, auc_baseline = calculate_PeakValue_and_AUC(sigs,
                                                                                       trial_info,
                                                                                       trial_type=trial_type,
                                                                                       baseline_window_start_time=baseline_window_start_time,
                                                                                       fs=fs, x_axis=x_axis,
                                                                                       fixed_trapz_start=trapz_start,
                                                                                       fixed_trapz_duration=trapz_duration,
                                                                                       fixed_auc_window=fixed_auc_window,
                                                                                       spout_key_times=spout_key_times)  # idx= 1, 2, 3

            output_dict.update({trial_type: (trial_info, auc_response, peak, auc_baseline)})

            # except ValueError:
            #     print()

        format_ax(ax)

        ax.set_xlabel("Time from trial onset (s)")
        ax.set_ylabel(r'($\Delta$F/F z-score)')

        # Might want to make this a variable
        ax.set_ylim([-5, 10])

        labels = [h.get_label() for h in legend_handles]

        fig.legend(handles=legend_handles, labels=labels, frameon=False, numpoints=1, bbox_to_anchor=[0.95, 0.95])

        fig.tight_layout()

        # plt.show()
        pdf.savefig()
        plt.close()

    # Write csv with area under curves
    with open(sep.join([output_path, recording_id + '_trialSignals.csv']), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        writer.writerow(['Recording'] + ['Trial_type'] + ['TrialID'] + ['AMdepth'] +
                        ['Trial_onset'] + ['Trial_offset'] + ['Area_under_curve'] + ['Peak_value'] + [
                            'Baseline_area_under_curve'])

        for trial_type in output_dict.keys():
            for trial_idx in range(len(output_dict[trial_type][0])):
                # output_list[x][0] is (cur_trial['TrialID'], cur_trial['AMdepth'], cur_trial['Trial_onset'])

                trialID = output_dict[trial_type][0][trial_idx][0]
                AMdepth = output_dict[trial_type][0][trial_idx][1]
                trial_onset = output_dict[trial_type][0][trial_idx][2]
                trial_offset = output_dict[trial_type][0][trial_idx][3]
                writer.writerow([recording_id] + [trial_type] + [trialID] + [np.round(AMdepth, 2)] +
                                [trial_onset] +  # Trial onset
                                [trial_offset] +
                                [output_dict[trial_type][1][trial_idx]] +  # Trapz
                                [output_dict[trial_type][2][trial_idx]] +  # Peak
                                [output_dict[trial_type][3][trial_idx]])  # Baseline AUC for dprime calculations
    #
    #

# SIGNALS_PATH = ''
# KEYS_PATH = ''
# OUTPUT_PATH = ''
#
# BASELINE_DURATION_FOR_ZSCORE = np.NaN  # in seconds; for firing rate calculation to non-AM trials
# STIM_DURATION_FOR_ZSCORE = np.NaN  # in seconds; for firing rate calculation to AM trials
#
#
# def run(main_path, baseline_duration_for_zscore, stim_duration_for_zscore, number_of_cores=int(cpu_count() / 2),
#         subjects_to_run=None, sessions_to_run=None):
#     warnings.filterwarnings("ignore")
#
#     BASELINE_DURATION_FOR_ZSCORE = baseline_duration_for_zscore  # in seconds; for firing rate calculation to non-AM trials
#     STIM_DURATION_FOR_ZSCORE = stim_duration_for_zscore  # in seconds; for firing rate calculation to AM trials
#
#     # number_of_cores  = int(cpu_count()/2)
#
#     number_of_cores = number_of_cores
#     # Only run these or None to run all
#     subjects_to_run = subjects_to_run
#     sessions_to_run = sessions_to_run
#
#     SIGNALS_PATH = '.' + sep + sep.join([main_path, 'Data', 'Whole session signal'])
#     KEYS_PATH = '.' + sep + sep.join([main_path, 'Data', 'Key files'])
#     OUTPUT_PATH = '.' + sep + sep.join([main_path, 'Data', 'Output'])
#
#     # Load existing JSONs; will be empty if this is the first time running
#     all_json = glob(OUTPUT_PATH + sep + 'JSON files' + sep + '*json')
#
#     # Generate a list of inputs to be passed to each worker
#     input_lists = list()
#     SIGNALS_PATH = glob(SIGNALS_PATH + sep + '*Aversive*dff.csv')
#     for dummy_idx, memory_path in enumerate(SIGNALS_PATH):
#
#         if subjects_to_run is not None:
#             if any([chosen for chosen in subjects_to_run if chosen in memory_path]):
#                 pass
#             else:
#                 continue
#
#         if sessions_to_run is not None:
#             if any([chosen for chosen in sessions_to_run if chosen in memory_path]):
#                 pass
#             else:
#                 continue
#
#         input_lists.append((memory_path, all_json))
#
#         # For debugging
#         # run_pipeline((memory_path, all_json))
#
#     pool = Pool(number_of_cores)
#
#     pool_map_result = pool.map(_run_pipeline, input_lists)
#
#     pool.close()
#
#     pool.join()
#
