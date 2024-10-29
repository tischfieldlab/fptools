import os

import matplotlib.pyplot as plt
import numpy as np

from fptools.io import Session, Signal
from fptools.preprocess.lib import detrend_double_exponential, estimate_motion, trim_signals


def dxp_motion_dff(session: Session, block, show_steps=True, plot_dir=''):
    try:
        if show_steps:
            fig, axs = plt.subplots(6, 1, figsize=(24, 6*6))

        sampling_rate = block.streams['_465A'].fs
        dopa = block.streams['_465A'].data
        isob = block.streams['_415A'].data
        time = np.linspace(1,len(dopa), len(dopa)) / sampling_rate

        if show_steps:
            axs[0].plot(time, dopa, label='Dopamine', c='g')
            axs[0].plot(time, isob, label='Isosbestic', c='r')
            axs[0].set_title('Raw signal')
            axs[0].legend()

        # trim raw signal start to when the optical system came online
        dopa_trimmed, isob_trimmed, time_trimmed = trim_signals(dopa, isob, time, begin=int(block.scalars.Fi1i.ts[0] * sampling_rate))

        if show_steps:
            axs[1].plot(time_trimmed, dopa_trimmed, label='Dopamine', c='g')
            axs[1].plot(time_trimmed, isob_trimmed, label='Isosbestic', c='r')
            axs[1].set_title('Trimmed Raw signal')
            axs[1].legend()

        # detrend using a double exponential fit
        dopa_detrend, dopa_fit = detrend_double_exponential(time_trimmed, dopa_trimmed)
        isob_detrend, isob_fit = detrend_double_exponential(time_trimmed, isob_trimmed)

        if show_steps:
            axs[2].plot(time_trimmed, dopa_trimmed, label='Dopamine', c='g')
            axs[2].plot(time_trimmed, dopa_fit, label='Dbl Exp Fit', c='k')
            axs[2].plot(time_trimmed, isob_trimmed, label='Isosbestic', c='r')
            axs[2].plot(time_trimmed, isob_fit, label='Dbl Exp Fit', c='k')
            axs[2].set_title('Double Exponential Fit')
            axs[2].legend()

            axs[3].plot(time_trimmed, dopa_detrend, label='Dopamine', c='g')
            axs[3].plot(time_trimmed, isob_detrend, label='Isosbestic', c='r')
            axs[3].set_title('De-trended signals')
            axs[3].legend()

        # correct for motion artifacts
        dopa_motion_corrected, est_motion = estimate_motion(dopa_detrend, isob_detrend)

        if show_steps:
            axs[4].plot(time_trimmed, dopa_motion_corrected, label='Dopamine', c='g')
            axs[4].plot(time_trimmed, est_motion, label='Estimated Motion', c='b')
            axs[4].set_title('Motion Correction')
            axs[4].legend()

        #dopa, isob = zscore_signals(dopa, isob)
        # computed deltaF / F
        dopa_norm = 100 * dopa_motion_corrected / dopa_fit

        if show_steps:
            axs[5].plot(time_trimmed, dopa_norm, label='Dopamine', c='g')
            axs[5].set_title('Normalized')
            axs[5].legend()

        session.signals['Dopamine'] = Signal('Dopamine', dopa_norm, time=time_trimmed, units='ΔF/F')
        session.signals['Isosbestic'] = Signal('Isosbestic', isob_detrend, time=time_trimmed, units='detrended mV')


        return session
    except:
        raise
    finally:
        if show_steps:
            fig.savefig(os.path.join(plot_dir, f'{block.info.blockname}.png'), dpi=600)
            fig.savefig(os.path.join(plot_dir, f'{block.info.blockname}.pdf'))
            plt.close(fig)
