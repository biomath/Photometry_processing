from multiprocessing import Pool, cpu_count
from warnings import filterwarnings
from glob import glob
from platform import system
from os.path import sep

from FP_get_trial_signals import run_pipeline as trial_signals_pipeline
from FP_get_signalsByAM import run_pipeline as signalsByAm_pipeline
from plot_behavioralDprimes import run_pipeline as plot_behavioralDprime

# Tweak the regex file separator for cross-platform compatibility
if system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


def run_trialSignals_pipeline(main_path, baseline_duration_for_zscore, stim_duration_for_zscore, number_of_cores=int(cpu_count() / 2),
                              subjects_to_run=None, sessions_to_run=None):
    # filterwarnings("ignore")

    BASELINE_DURATION_FOR_ZSCORE = baseline_duration_for_zscore  # in seconds; for firing rate calculation to non-AM trials
    STIM_DURATION_FOR_ZSCORE = stim_duration_for_zscore  # in seconds; for firing rate calculation to AM trials

    # number_of_cores  = int(cpu_count()/2)

    number_of_cores = number_of_cores
    # Only run these or None to run all
    subjects_to_run = subjects_to_run
    sessions_to_run = sessions_to_run

    SIGNALS_PATH = '.' + sep + sep.join([main_path, 'Whole session signal'])
    SIGNALS_PATH = glob(SIGNALS_PATH + sep + '*Aversive*dff.csv')
    KEYS_PATH = '.' + sep + sep.join([main_path, 'Key files'])
    OUTPUT_PATH = '.' + sep + sep.join([main_path, 'Output'])

    globals_input_list = (BASELINE_DURATION_FOR_ZSCORE, STIM_DURATION_FOR_ZSCORE, KEYS_PATH, OUTPUT_PATH)

    # Load existing JSONs; will be empty if this is the first time running
    all_json = glob(OUTPUT_PATH + sep + 'JSON files' + sep + '*json')

    # Generate a list of inputs to be passed to each worker
    file_input_lists = list()

    DEBUG = True
    for dummy_idx, memory_path in enumerate(SIGNALS_PATH):

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

    BASELINE_DURATION_FOR_ZSCORE = baseline_duration_for_zscore  # in seconds; for firing rate calculation to non-AM trials
    STIM_DURATION_FOR_ZSCORE = stim_duration_for_zscore  # in seconds; for firing rate calculation to AM trials

    # number_of_cores  = int(cpu_count()/2)

    number_of_cores = number_of_cores
    # Only run these or None to run all
    subjects_to_run = subjects_to_run
    sessions_to_run = sessions_to_run

    SIGNALS_PATH = '.' + sep + sep.join([main_path, 'Whole session signal'])
    SIGNALS_PATH = glob(SIGNALS_PATH + sep + '*Aversive*dff.csv')
    KEYS_PATH = '.' + sep + sep.join([main_path, 'Key files'])
    OUTPUT_PATH = '.' + sep + sep.join([main_path, 'Output'])

    globals_input_list = (BASELINE_DURATION_FOR_ZSCORE, STIM_DURATION_FOR_ZSCORE, KEYS_PATH, OUTPUT_PATH)

    # Load existing JSONs; will be empty if this is the first time running
    all_json = glob(OUTPUT_PATH + sep + 'JSON files' + sep + '*json')

    # Generate a list of inputs to be passed to each worker
    file_input_lists = list()

    for dummy_idx, memory_path in enumerate(SIGNALS_PATH):

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
                              subjects_to_run=None, sessions_to_run=None):
    # filterwarnings("ignore")
    # number_of_cores  = int(cpu_count()/2)

    number_of_cores = number_of_cores
    # Only run these or None to run all
    subjects_to_run = subjects_to_run
    sessions_to_run = sessions_to_run

    KEYS_PATH = '.' + sep + sep.join([main_path, 'Key files'])
    dprimeMat_files = glob(KEYS_PATH + sep + '*_dprimeMat.csv')
    OUTPUT_PATH = '.' + sep + sep.join([main_path, 'Output'])

    globals_input_list = (KEYS_PATH, OUTPUT_PATH)

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
    AAVrg-GCaMP8m in ACx
    Fiber in VO
    '''
    main_path = 'Data_AAVrg-GCaMP8s-ACx_fiber-VO'
    baseline_duration_for_zscore = 1
    stim_duration_for_zscore = 5  # Does not affect AUC calculation
    # sessions_to_run = ['SUBJ-ID-336_FP-Aversive-AM-220427-143620_dff']  # GRC22 poster
    subjects_to_run = ['SUBJ-ID-336']
    run_trialSignals_pipeline(main_path, baseline_duration_for_zscore, stim_duration_for_zscore, subjects_to_run=subjects_to_run)
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