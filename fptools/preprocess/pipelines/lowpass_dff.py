import os

import matplotlib.pyplot as plt

from fptools.preprocess.lib import lowpass_filter, trim_signals, fs2t, downsample as downsample_fn
from fptools.io import Session, Signal


def lowpass_dff(session: Session, block, show_steps=True, plot_dir='', downsample=None):
    try:
        if show_steps:
            fig, axs = plt.subplots(4, 1, figsize=(24, 6*4))

        sampling_rate = block.streams['_465A'].fs
        dopa = block.streams['_465A'].data
        isob = block.streams['_415A'].data
        time = fs2t(sampling_rate, len(dopa))

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


        dopa_lowpass = lowpass_filter(dopa_trimmed, sampling_rate, 0.01)
        isob_lowpass = lowpass_filter(isob_trimmed, sampling_rate, 0.01)

        if show_steps:
            axs[2].plot(time_trimmed, dopa_lowpass, label='Dopamine', c='g')
            axs[2].plot(time_trimmed, isob_lowpass, label='Isosbestic', c='r')
            axs[2].set_title('Lowpasss filtered (0.01 Hz)')
            axs[2].legend()

        dopa_dff = ((dopa_trimmed - dopa_lowpass) / dopa_lowpass) * 100
        isob_dff = ((isob_trimmed - isob_lowpass) / isob_lowpass) * 100

        if show_steps:
            axs[3].plot(time_trimmed, dopa_dff, label='Dopamine', c='g')
            axs[3].plot(time_trimmed, isob_dff, label='Isosbestic', c='r')
            axs[3].set_title('Normalized (dff)')
            axs[3].legend()

        if downsample is not None:
            time_trimmed, dopa_dff, isob_dff = downsample_fn(time_trimmed, dopa_dff, isob_dff, factor=downsample)


        session.signals['Dopamine'] = Signal('Dopamine', dopa_dff, time=time_trimmed, units='ΔF/F')
        session.signals['Isosbestic'] = Signal('Isosbestic', isob_dff, time=time_trimmed, units='ΔF/F')

        return session
    except:
        raise
    finally:
        if show_steps:
            fig.savefig(os.path.join(plot_dir, f'{block.info.blockname}.png'), dpi=600)
            fig.savefig(os.path.join(plot_dir, f'{block.info.blockname}.pdf'))
            plt.close(fig)
