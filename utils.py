import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
from pypianoroll import Multitrack, Track


def grid_plot(ppr, bar_range=None, pitch_range='auto', beats_in_bar=4, beat_resolution=24, figsize=[21, 10]):
    """
    pretty ploting for pypianoroll
    """
    orgSize = rcParams['figure.figsize']
    rcParams['figure.figsize'] = figsize
    
    if isinstance(ppr, Track):
        ppr = Multitrack(tracks=[ppr], downbeat=list(range(ppr.pianoroll.shape[0])), beat_resolution=beat_resolution)
    
    beat_res = ppr.beat_resolution
    bar_res = beats_in_bar * beat_res
    downbeat = ppr.downbeat
    ppr.downbeat = np.zeros_like(ppr.downbeat, dtype=bool)
    
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    major_color = ['red', 'orange', 'yellow', 'green', 'cyan', 'mediumblue', 'magenta']
    major = list(zip(major_scale, major_color))
    
    fig, axs = ppr.plot(xtick="beat")
    for a, ax in enumerate(axs):
        ax.grid(False)
        ax.xaxis.set_ticks_position('none')
        ax.set_xticklabels([], minor=True)
        ax.set_xticklabels(range(len(ppr.downbeat) // beat_res), minor=False)
        
        xlim = ax.get_xlim()
        if bar_range:
            xlim = (bar_range[0] * bar_res, bar_range[1] * bar_res - 0.5)
        ax.set_xlim(*xlim)
        
        if pitch_range == 'auto':
            try:
                low, high = ppr.tracks[a].get_active_pitch_range()
            except ValueError:
                low, high = 66, 66
            ax.set_ylim(max(0, low - 6), min(high + 6, 127))
        elif pitch_range:
            pr = np.array(pitch_range)
            if pr.ndim == 1:
                ax.set_ylim(pr[0], pr[1])
            elif pr.ndim == 2:
                ax.set_ylim(pr[a][0], pr[a][1])
        ylim = ax.get_ylim()
                
        for bar_step in range(int(xlim[0]), int(xlim[1])+1, bar_res):
            ax.vlines(bar_step - 0.5, 0, 127)
            for beat in range(1, 4):
                ax.vlines(bar_step + beat_res * beat - 0.5, 0, 127, linestyles='dashed')

        for k, color in major:
            linewidth = 2.0 if k == 0 else 1.0
            for h in range(int(ylim[0]), int(ylim[1])):
                if h % 12 == k:
                    ax.hlines(h, xlim[0], xlim[1], linestyles='-', linewidth=linewidth, color=color)
    
    ppr.downbeat = downbeat
    
    rcParams['figure.figsize'] = orgSize


import time
class Timer():
    """
    with Timer():
        # 計測したい処理
        # 約 1/100000 [sec] だけこいつを使った方が遅くなることに注意
    
    with Timer(fmt="endtime: {:f}"):
        # 計測したい処理
        # このようにformatを指定することもできる
    
    """
    def __init__(self, fmt='{:f}'):
        self.fmt = fmt
    
    def get_time():
        return time.time() - self.start
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, _1, _2, _3):
        end = time.time() - self.start
        print(self.fmt.format(end))
