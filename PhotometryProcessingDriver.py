from multiprocessing import Pool, cpu_count
from warnings import filterwarnings
from glob import glob
from platform import system
from os.path import sep
from os import makedirs

from FP_get_trial_signals import run_pipeline as trial_signals_pipeline
from FP_get_signalsByAM import run_pipeline as signalsByAm_pipeline
from plot_behavioralDprimes import run_pipeline as plot_behavioralDprime

# Tweak the regex file separator for cross-platform compatibility
if system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


def run_trialSignals_pipeline(main_path, baseline_duration_for_zscore, stim_duration_for_zscore, number_of_cores=int(cpu_count() / 2),
                              subjects_to_run=None, sessions_to_run=None, fixed_auc_window=True):
    # filterwarnings("ignore")

    baseline_duration_for_zscore = baseline_duration_for_zscore  # in seconds; for firing rate calculation to non-AM trials
    stim_duration_for_zscore = stim_duration_for_zscore  # in seconds; for firing rate calculation to AM trials

    # number_of_cores  = int(cpu_count()/2)

    number_of_cores = number_of_cores
    # Only run these or None to run all
    subjects_to_run = subjects_to_run
    sessions_to_run = sessions_to_run

    signals_path = '.' + sep + sep.join([main_path, 'Whole session signal'])
    signals_path = glob(signals_path + sep + '*Aversive*dff.csv')
    keys_path = '.' + sep + sep.join([main_path, 'Key files'])
    output_path = '.' + sep + sep.join([main_path, 'Output', 'SpoutOffset centered'])
    makedirs(output_path, exist_ok=True)

    globals_input_list = (baseline_duration_for_zscore, stim_duration_for_zscore, keys_path, output_path, fixed_auc_window)

    # Load existing JSONs; will be empty if this is the first time running
    all_json = glob(output_path + sep + 'JSON files' + sep + '*json')

    # Generate a list of inputs to be passed to each worker
    file_input_lists = list()

    DEBUG = False
    for dummy_idx, memory_path in enumerate(signals_path):

        if subjects_to_run is not None:
            if any([chosen for chosen in subjects_to_run if chosen in memory_path]):
                pass
            else:
                continue

        if sessions_to_run is not None:
            if any([chosen for chosen in sessions_to_run if chosen in memory_path]):
                pass
            else:
                continue

        if not DEBUG:
            file_input_lists.append((globals_input_list, (memory_path, all_json)))
        #
        # For debugging
        if DEBUG:
            trial_signals_pipeline((globals_input_list, (memory_path, all_json)))

    if not DEBUG:
        pool = Pool(number_of_cores)
        #
        pool.map(trial_signals_pipeline, file_input_lists)
        #
        pool.close()
        #
        pool.join()

def run_signalsByAM_pipeline(main_path, baseline_duration_for_zscore, stim_duration_for_zscore, number_of_cores=int(cpu_count() / 2),
                              subjects_to_run=None, sessions_to_run=None):
    # filterwarnings("ignore")

    baseline_duration_for_zscore = baseline_duration_for_zscore  # in seconds; for firing rate calculation to non-AM trials
    stim_duration_for_zscore = stim_duration_for_zscore  # in seconds; for firing rate calculation to AM trials

    # number_of_cores  = int(cpu_count()/2)

    number_of_cores = number_of_cores
    # Only run these or None to run all
    subjects_to_run = subjects_to_run
    sessions_to_run = sessions_to_run

    signals_path = '.' + sep + sep.join([main_path, 'Whole session signal'])
    signals_path = glob(signals_path + sep + '*Aversive*dff.csv')
    keys_path = '.' + sep + sep.join([main_path, 'Key files'])
    output_path = '.' + sep + sep.join([main_path, 'Output'])

    globals_input_list = (baseline_duration_for_zscore, stim_duration_for_zscore, keys_path, output_path)

    # Load existing JSONs; will be empty if this is the first time running
    all_json = glob(output_path + sep + 'JSON files' + sep + '*json')

    # Generate a list of inputs to be passed to each worker
    file_input_lists = list()

    for dummy_idx, memory_path in enumerate(signals_path):

        if subjects_to_run is not None:
            if any([chosen for chosen in subjects_to_run if chosen in memory_path]):
                pass
            else:
                continue

        if sessions_to_run is not None:
            if any([chosen for chosen in sessions_to_run if chosen in memory_path]):
                pass
            else:
                continue

        signalsByAm_pipeline((globals_input_list, (memory_path, all_json)))


def run_plot_behavioralDprimes(main_path, baseline_duration_for_zscore, stim_duration_for_zscore, number_of_cores=int(cpu_count() / 2),
                              subjects_to_run=None, sessions_to_run=None, fixed_auc_window=True):
    # filterwarnings("ignore")
    # number_of_cores  = int(cpu_count()/2)

    number_of_cores = number_of_cores
    # Only run these or None to run all
    subjects_to_run = subjects_to_run
    sessions_to_run = sessions_to_run

    keys_path = '.' + sep + sep.join([main_path, 'Key files'])
    dprimeMat_files = glob(keys_path + sep + '*_dprimeMat.csv')
    output_path = '.' + sep + sep.join([main_path, 'Output'])
    FIXED_AUC_WINDOW = fixed_auc_window

    globals_input_list = (keys_path, output_path)

    for dummy_idx, file_path in enumerate(dprimeMat_files):

        if subjects_to_run is not None:
            if any([chosen for chosen in subjects_to_run if chosen in file_path]):
                pass
            else:
                continue

        plot_behavioralDprime((globals_input_list, (file_path,)))

if __name__ == '__main__':
    filterwarnings("ignore")
    '''
    AAVrg-GCaMP8s in ACx
    Fiber in VO
    '''
    main_path = 'Data_AAVrg-GCaMP8s-ACx_fiber-VO'
    baseline_duration_for_zscore = 1
    stim_duration_for_zscore = 5  # Does not affect AUC calculation
    fixed_auc_window = False
    # sessions_to_run = ['SUBJ-ID-336_FP-Aversive-AM-220427-143620_dff']  # GRC22 poster
    subjects_to_run = None
    run_trialSignals_pipeline(main_path, baseline_duration_for_zscore, stim_duration_for_zscore, subjects_to_run=subjects_to_run,
                              fixed_auc_window=fixed_auc_window)
    # run_signalsByAM_pipeline(main_path, baseline_duration_for_zscore, stim_duration_for_zscore,
    #                           subjects_to_run=subjects_to_run)

    '''
    AAVrg-EGFP in ACx
    Fiber in VO
    '''
    # main_path = 'Data_AAVrg-EGFP-ACx_fiber_VO'
    # baseline_duration_for_zscore = 1
    # stim_duration_for_zscore = 5
    # run(main_path, baseline_duration_for_zscore, stim_duration_for_zscore)

    '''
    AAV1-GCaMP8m in LO
    Fiber in LO
    '''
    # main_path = 'Data_AAV1-GCaMP8m-LO_fiber-LO'
    # baseline_duration_for_zscore = 1
    # stim_duration_for_zscore = 5
    # run(main_path, baseline_duration_for_zscore, stim_duration_for_zscore)