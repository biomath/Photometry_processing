from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from re import split, search
from glob import glob
from os.path import sep
from os import remove, makedirs
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
import json
import platform
from datetime import datetime
import warnings
from multiprocessing import Pool, cpu_count, current_process
from format_axes import format_ax

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep

def run_pipeline(input_list):
    memory_path, all_json = input_list
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

    key_paths_info = glob(KEYS_PATH + sep + subject_id + '*' +
                          cur_date + '-' + cur_timestamp + "*_trialInfo.csv")
    key_paths_spout = glob(KEYS_PATH + sep + subject_id + '*' +
                           cur_date + '-' + cur_timestamp + "*spoutTimestamps.csv")

    if len(key_paths_info) == 0:
        print("Key not found for " + recording_id)
        return

    # Load key files
    info_key_times = pd.read_csv(key_paths_info[0])

    # Load signal
    processed_signal = pd.read_csv(memory_path)

    # Grab trial times
    trial_key_times = info_key_times[info_key_times['TrialType'] == 0]
    trial_key_times = trial_key_times[(trial_key_times['Trial_onset'] >= T1) & (trial_key_times['Trial_onset'] < T2)]

    snippet = processed_signal[(processed_signal['Time'] >= T1) & (processed_signal['Time'] < T2)]

    with PdfPages(sep.join([OUTPUT_PATH, recording_id + '_signalSnippet.pdf'])) as pdf:
        fig, ax = plt.subplots(1, 1)

        # color_405 = '#AE5194'
        # color_465 = '#51AE6B'
        color_465 = 'black'

        # Plot shadings first and change alpha according to AM depth

        # Color Hits and Misses differently
        hit_color = '#60B2E5'
        missShock_color ='#C84630'
        missNoShock_color = '#F0A202'
        # for _, key_time in trial_key_times.iterrows():
        #     cur_amdepth = np.round(key_time['AMdepth'], 2)
        #     if key_time['Hit'] == 1:
        #         cur_color = hit_color
        #     elif key_time['ShockFlag'] == 1:
        #         cur_color = missShock_color
        #     else:
        #         cur_color = missNoShock_color
        #     cur_alpha = 0.5
        #     ax.axvspan(xmin=key_time['Trial_onset']-T1, xmax=key_time['Trial_offset']-T1,
        #                ymin=0.2, ymax=0.5, facecolor=cur_color, alpha=cur_alpha)
        #
        # legend_list = list()
        # # for cur_amdepth in sorted(list(set(trial_key_times['AMdepth'])), reverse=True):
        # #     alpha = 0.3
        # #     legend_list.append(patches.Patch(facecolor='red', edgecolor=None, alpha=alpha,
        # #                                      label=str(np.round(cur_amdepth, 2)*100) + '% depth'))
        # #
        # legend_list.append(patches.Patch(facecolor=hit_color, edgecolor=None, alpha=cur_alpha,
        #                                      label='AM trial - Hit'))
        # legend_list.append(patches.Patch(facecolor=missNoShock_color, edgecolor=None, alpha=cur_alpha,
        #                                      label='AM trial - Miss (no shock)'))
        # legend_list.append(patches.Patch(facecolor=missShock_color, edgecolor=None, alpha=cur_alpha,
        #                                      label='AM trial - Miss (shock)'))

        # Plot 405 first
        # ax.plot(snippet['Time']-T1, snippet['Ch405_mV'], color=color_405, alpha=0.8)
        # ax.plot(snippet['Time'] - T1, snippet['Ch465_mV'], color=color_465, alpha=0.8)

        ax.plot(snippet['Time']-T1, snippet['Ch465_dff'], color=color_465, alpha=0.8)

        format_ax(ax)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r'Normalized fluorescence ($\Delta$F/F)')

        # labels = [h.get_label() for h in legend_list]
        # fig.legend(handles=legend_list, labels=labels, frameon=False, numpoints=1, fontsize=rcParams['font.size'])

        # ax.set_ylim([-3, 8])
        ax.set_ylim([-20, 20])
        # plt.show()
        pdf.savefig()
        plt.close()




"""
Set global paths and variables
"""

# Set plotting parameters
LABEL_FONT_SIZE = 15
TICK_LABEL_SIZE = 10
rcParams['figure.figsize'] = (12, 10)
rcParams['figure.dpi'] = 300

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'Arial'
rcParams['font.weight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'

rcParams['font.size'] = LABEL_FONT_SIZE * 1.5
rcParams['axes.labelsize'] = LABEL_FONT_SIZE * 1.5
rcParams['axes.titlesize'] = LABEL_FONT_SIZE
rcParams['axes.linewidth'] = LABEL_FONT_SIZE / 12.
rcParams['legend.fontsize'] = LABEL_FONT_SIZE / 2.
rcParams['xtick.labelsize'] = TICK_LABEL_SIZE * 1.5
rcParams['ytick.labelsize'] = TICK_LABEL_SIZE * 1.5
rcParams['errorbar.capsize'] = LABEL_FONT_SIZE
rcParams['lines.markersize'] = LABEL_FONT_SIZE / 30.
rcParams['lines.markeredgewidth'] = LABEL_FONT_SIZE / 30.
rcParams['lines.linewidth'] = LABEL_FONT_SIZE / 8.


warnings.filterwarnings("ignore")

# main_path = 'Data_AAVrg-EGFP-ACx_fiber_VO'
main_path = 'Data_AAVrg-GCaMP8s-ACx_fiber-VO'
SIGNALS_PATH = '.' + sep + sep.join([main_path, 'Whole session signal'])
KEYS_PATH = '.' + sep + sep.join([main_path, 'Key files'])
OUTPUT_PATH = '.' + sep + sep.join([main_path, 'Output'])

# NUMBER_OF_CORES = int(cpu_count()/2)
#
NUMBER_OF_CORES = 1
# # Only run these su or None to run all
# SESSION_TO_RUN = ['SUBJ-ID-108_FP-Aversive-AM-210629-104630_dff.csv']
# T1 = 500
# T2 = 580
# CSV_PRENAME = 'ACx-GFP_hSyn_sigSnippet_longer'

#
SESSION_TO_RUN = ['SUBJ-ID-336_FP-Aversive-AM-220421-144912']

# 100 s plotted for the LSRF grant
# 80 s plotted for MLC 211227
T1 = 0
T2 = np.Inf


if __name__ == '__main__':
    # Load existing JSONs; will be empty if this is the first time running
    all_json = glob(OUTPUT_PATH + sep + 'JSON files' + sep + '*json')

    # Generate a list of inputs to be passed to each worker
    input_lists = list()
    signal_paths = glob(SIGNALS_PATH + sep + '*dff.csv')
    for dummy_idx, memory_path in enumerate(signal_paths):

        if SESSION_TO_RUN is not None:
            if any([chosen for chosen in SESSION_TO_RUN if chosen in memory_path]):
                pass
            else:
                continue

        input_lists.append((memory_path, all_json))
        run_pipeline((memory_path, all_json))

        # run_pipeline((memory_path, all_json))
    # pool = Pool(NUMBER_OF_CORES)
    #
    #
    # # Feed each worker with all memory paths from one unit
    # pool_map_result = pool.map(run_pipeline, input_lists)
    #
    # pool.close()
    # #
    # # if pool_map_result.ready():
    # pool.join()

