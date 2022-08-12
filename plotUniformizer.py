from matplotlib import rcParams
import runpy


if __name__ == "__main__":
    # label_font_size = 5
    # tick_label_size = 5
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
    # rcParams['errorbar.capsize'] = label_font_size/2
    # rcParams['lines.markersize'] = line_thickness/2
    # rcParams['lines.linewidth'] = line_thickness/2
    #
    # rcParams['figure.figsize'] = (1.5, 1.5)
    # runpy.run_module(mod_name='PhotometryProcessingDriver', run_name='__main__')


    # For GRC22 poster
    label_font_size = 23
    tick_label_size = 15
    legend_font_size = 15
    line_thickness = 2

    rcParams['figure.dpi'] = 600
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams['font.family'] = 'Arial'

    rcParams['font.size'] = label_font_size
    rcParams['axes.labelsize'] = label_font_size
    rcParams['axes.titlesize'] = label_font_size
    rcParams['axes.linewidth'] = line_thickness
    rcParams['legend.fontsize'] = legend_font_size
    rcParams['xtick.labelsize'] = tick_label_size
    rcParams['ytick.labelsize'] = tick_label_size
    rcParams['errorbar.capsize'] = label_font_size/2
    rcParams['lines.markersize'] = line_thickness/2
    rcParams['lines.linewidth'] = line_thickness/2

    rcParams['figure.figsize'] = (5, 5)
    runpy.run_module(mod_name='PhotometryProcessingDriver', run_name='__main__')