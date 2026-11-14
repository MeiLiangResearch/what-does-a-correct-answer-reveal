"""Shared Seaborn typography and line widths for the current manuscript figures."""
import seaborn as sns
import matplotlib.pyplot as plt

AXIS_WIDTH = 0.8
DATA_WIDTH = 1.2
ERROR_WIDTH = 1.1
INK = '#263744'
MUTED = '#61717A'


def apply_style():
    sns.set_theme(style='ticks', context='paper', font='Arial', rc={
        'font.size': 9.5, 'axes.labelsize': 9.5, 'axes.titlesize': 10.5,
        'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 9.5,
        'axes.linewidth': AXIS_WIDTH, 'xtick.major.width': AXIS_WIDTH,
        'ytick.major.width': AXIS_WIDTH, 'xtick.minor.width': AXIS_WIDTH,
        'ytick.minor.width': AXIS_WIDTH, 'xtick.major.size': 3,
        'ytick.major.size': 3, 'lines.linewidth': DATA_WIDTH,
        'lines.markeredgewidth': AXIS_WIDTH, 'axes.edgecolor': '#52636D',
        'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
        'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def save_figure(fig, folder):
    """Keep SVG text editable and trim only the empty exterior canvas."""
    folder.mkdir(parents=True, exist_ok=True)
    settings = dict(bbox_inches='tight', pad_inches=0.035, facecolor='white')
    fig.savefig(folder / 'figure.svg', **settings)
    fig.savefig(folder / 'figure.pdf', **settings)
    fig.savefig(folder / 'figure.png', dpi=300, **settings)
    fig.savefig(folder / 'figure.jpg', dpi=600, **settings)
    plt.close(fig)
