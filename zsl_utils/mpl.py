import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from matplotlib import rcParams
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import Normalize, SymLogNorm

import copy
import json
from pathlib import Path
import PIL

rcParams['font.family'] = 'serif'
rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 12
rcParams['axes.titlesize'] = 16
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = '\\usepackage{amssymb} \\usepackage{amsmath}'

FIGSIZE_NARROW = (3,5)
FIGSIZE_MED = (6,5)
FIGSIZE_WIDE = (9,5)

def get_color(v, vmin=0, vmax=1, cmap_key: str = 'plasma'):
    cmap = plt.get_cmap(cmap_key)
    norm = Normalize(vmin=vmin, vmax=vmax)
    color = cmap(norm(v))
    return color
    

def save_figure(fig, figpath: Path, save_png: bool = True, save_json: bool = True):
    fig.savefig(figpath.with_suffix('.pdf'), format='pdf')

    if save_png:
        fig.savefig(figpath.with_suffix('.png'), format='png')

    if save_json:
        json_dict = {}
        for i, ax in enumerate(fig.axes):
            ax_key = f"ax_{i}"
            json_dict[ax_key] = {}
            for line in ax.get_lines():
                label = line.get_label()
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                if label and len(xdata) and not label.startswith('_'):
                    json_dict[ax_key][label] = {'x': xdata.tolist(), 'y': ydata.tolist()}
        figpath.with_suffix('.json').write_text(single_line_list_json_dumps(json_dict, indent=2))
        


def single_line_list_json_dumps(d: dict, indent=2):
    def mark_list(dict_or_list):
        if isinstance(dict_or_list, list):
            if isinstance(dict_or_list[0], (list, dict)):
                for i,x in enumerate(dict_or_list):
                    dict_or_list[i] = mark_list(x)
            elif isinstance(dict_or_list[0], (int, float, str)):
                return "##<{}>##".format(dict_or_list)
            else:
                raise Exception(f"Unexpected type {type(dict_or_list[0])}")
        elif isinstance(dict_or_list, dict):
            for k,v in dict_or_list.items():
                if isinstance(v, (list, dict)):
                    dict_or_list[k] = mark_list(v)
            return dict_or_list
        else:
            raise Exception(f"Unexpected type {type(dict_or_list)}")

    _d = copy.deepcopy(d)
    mark_list(_d)
    json_str = json.dumps(_d, indent=indent)
    json_str = json_str.replace('"##<', '').replace('>##"', '')
    return json_str