import numpy as np
import matplotlib as mpl
#mpl.rcParams['animation.ffmpeg_path'] = '/mnt/home/ktavangar/ffmpeg/bin/ffmpeg.exe'
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from functools import partial

import sys, importlib
from helper import animate_m1m2

from astropy.table import Table

tbl = Table.read("/mnt/home/ktavangar/ceph/full_mssa_prep_table.fits")
tbl.sort(["timestep", "jphi_cen", "tphi_cen"])


# takes a couple minutes to run
fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(14, 4), 
                                    subplot_kw=dict(projection="polar"))

print('making animation...')
anim_m1m2 = FuncAnimation(fig,
                          partial(animate_m1m2, tbl=tbl, axs=[ax1, ax2, ax3]),
                          frames=np.arange(0, 600, 1),
                          interval=2,
                          blit=False)

print('saving animation...')

f = '../figures/m1m2_coeff_amp_ratio.mp4'
FFwriter = animation.FFMpegWriter(fps=10)
anim_m1m2.save(f, writer = FFwriter)