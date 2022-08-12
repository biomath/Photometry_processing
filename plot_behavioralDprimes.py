from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from scipy.integrate import simps
from re import split, search
from glob import glob
from os.path import sep
from os import remove, makedirs
import csv
import matplotlib.pyplot as plt
from matplotlib import patches
from seaborn import FacetGrid, distplot
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


def run_pipeline(input_list):
    (KEYS_PATH, OUTPUT_PATH), (file_path,) = input_list
    # Split path name to get subject, session and unit ID for prettier output
    split_file_path = split(REGEX_SEP, file_path)  # split path

    subject_id = split_file_path[0]

    # Load file
    dprime_file = pd.read_csv(file_path[0])

    # TODO: figure out how to fit the psignifit function

    with PdfPages(sep.join([OUTPUT_PATH, subject_id + '_behavioralDprimes.pdf'])) as pdf:
        for amdepth in sorted(list(set(all_ams))):
            # fig, ax = plt.subplots(1, 1)
            fig = plt.figure()

            ax = FacetGrid(dprime_file, hue='Block_id')
            ax = ax.map(distplot, "d_prime", hist=False, rug=True)

            format_ax(ax)

            ax.set_xlabel("Time from trial onset (s)")
            ax.set_ylabel(r'Normalized fluorescence ($\Delta$F/F z-score)')

            # Might want to make this a variable
            ax.set_ylim([-5, 10])

            labels = [h.get_label() for h in legend_handles]

            fig.legend(handles=legend_handles, labels=labels, frameon=False, numpoints=1)

            fig.suptitle(str(amdepth) + 'dB (re:100%)')

            fig.tight_layout()

            # plt.show()
            pdf.savefig()
            plt.close()