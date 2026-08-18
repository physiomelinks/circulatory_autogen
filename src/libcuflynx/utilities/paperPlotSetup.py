import matplotlib as mpl
import numpy as np

def Setup_Plot(case):
    #case=1 is for seperated column paper
    #case=2 is for full size paper
    #case=3 is for very large on page
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    # fontPath = os.path.join('C:\\', 'Users', 'fargus', 'Documents', 'pycharm_projects', 'xfoil', 'venv', 'Lib',
    #                         'site-packages', 'matplotlib', 'mpl-data', 'fonts', 'ttf', 'cmunrm.ttf')

    # if not os.path.exists(fontPath):
    #     print('font path doesnt exist')
    #     sys.exit()

    # prop = font_manager.FontProperties(fname=fontPath)
    # mpl.rcParams['font.family'] = prop.get_name()
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'cmr10'
    mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    mpl.rcParams['grid.color'] = 'xkcd:light grey'
    mpl.rcParams['legend.edgecolor'] = 'k'
    mpl.rcParams['legend.fancybox'] = False
    mpl.rcParams['legend.framealpha'] = 1.0
    # calculate figure width and height for number of columns
    assert(case in [1,2,3])
    if case == 1:
        fig_width = 3.8
        mpl.rcParams['font.size'] = 8
        mpl.rcParams['xtick.labelsize'] = 8
        mpl.rcParams['ytick.labelsize'] = 8
        mpl.rcParams['axes.labelsize'] = 8
        mpl.rcParams['legend.fontsize'] = 6
        mpl.rcParams['lines.linewidth'] = 0.8
    elif case ==2:
        fig_width = 6.9
        mpl.rcParams['font.size'] = 20
        mpl.rcParams['xtick.labelsize'] = 16
        mpl.rcParams['ytick.labelsize'] = 16
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['legend.fontsize'] = 20
        mpl.rcParams['lines.linewidth'] = 1.8
    elif(case ==3):
        fig_width = 17.4/2.54 #17.54 cm to inches
        mpl.rcParams['font.size'] = 14
        mpl.rcParams['xtick.labelsize'] = 12
        mpl.rcParams['ytick.labelsize'] = 12
        mpl.rcParams['axes.labelsize'] = 14
        mpl.rcParams['legend.fontsize'] = 12
        mpl.rcParams['lines.linewidth'] = 0.5


    # aesthetic ratio
    golden_mean = (np.sqrt(5) - 1.0)/2.0
    fig_height = fig_width * golden_mean
    mpl.rcParams['figure.figsize'] = [fig_width, fig_height]


