import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from fptools.preprocess.lib import lowpass_filter, trim_signals, fs2t, downsample as downsample_fn
from fptools.io import Session, Signal, SignalMapping


def lowpass_dff(session: Session, block: Any, signal_map: list[SignalMapping], show_steps: bool = True,
                plot_dir: str = '', downsample: Optional[int] = None) -> Session:
    '''A "simple" preprocess pipeline based on ultra-lowpass filtering.

    Implemented as described in:
    Cai, Kaeser, et al. Dopamine dynamics are dispensable for movement but promote reward responses.
    Nature, 2024. https://doi.org/10.1038/s41586-024-08038-z

    Pipeline steps:
    1) Signals are trimmed to the optical system start.
    2) Signals are lowpass filterd at 0.01 Hz (~100 second timescale)
    3) Signals are converted to dF/F using lowpass filtered signals as F0
    4) Signals are optionally downsampled factor `downsample`

    Parameters:
    session: the session to populate.
    block: block data struct from `tdt.read_block()`.
    signal_map: mapping of signals to perform
    show_steps: if `True`, produce diagnostic plots of the preprocessing steps.
    plot_dir: path where diagnostic plots of the preprocessing steps should be saved.
    downsample: if not `None`, downsample signal by `downsample` factor.
    '''
    try:
        if show_steps:
            fig, axs = plt.subplots(4, 1, figsize=(24, 6*4))
            palette = sns.color_palette('colorblind', n_colors=len(signal_map))

        signals: list[Signal] = []
        for sm in signal_map:
            stream = block.streams[sm['tdt_name']]
            signals.append(Signal(sm['dest_name'], stream.data, fs=stream.fs))

        if show_steps:
            for i, sig in enumerate(signals):
                axs[0].plot(sig.time, sig.signal, label=sig.name, c=palette[i])
            axs[0].set_title('Raw signal')
            axs[0].legend()

        # trim raw signal start to when the optical system came online
        for sig in signals:
            sig.signal, sig.time = trim_signals(sig.signal, sig.time, begin=int(block.scalars.Fi1i.ts[0] * sig.fs))

        if show_steps:
            for i, sig in enumerate(signals):
                axs[0].plot(sig.time, sig.signal, label=sig.name, c=palette[i])
            axs[1].set_title('Trimmed Raw signal')
            axs[1].legend()

        # lowpass filter at 0.01 Hz
        lowpass_signals = []
        for sig in signals:
            s = sig.copy()
            s.signal = lowpass_filter(sig.signal, sig.fs, 0.01)
            lowpass_signals.append(s)

        if show_steps:
            for i, sig in enumerate(lowpass_signals):
                axs[0].plot(sig.time, sig.signal, label=sig.name, c=palette[i])
            axs[2].set_title('Lowpasss filtered (0.01 Hz)')
            axs[2].legend()

        # calculate dF/F
        for sig, lowpass_sig in zip(signals, lowpass_signals):
            sig.signal = ((sig.signal - lowpass_sig.signal) / lowpass_sig.signal) * 100
            sig.units = 'ΔF/F'

        if show_steps:
            for i, sig in enumerate(signals):
                axs[0].plot(sig.time, sig.signal, label=sig.name, c=palette[i])
            axs[3].set_title('Normalized (dff)')
            axs[3].legend()

        # possibly downsample
        if downsample is not None:
            for sig in signals:
                sig.signal, sig.time = downsample_fn(sig.signal, sig.time, window=downsample, factor=downsample)

        # construct Signals and add to the Session
        for sig in signals:
            session.add_signal(sig)

        return session
    except:
        raise
    finally:
        if show_steps:
            fig.savefig(os.path.join(plot_dir, f'{block.info.blockname}.png'), dpi=600)
            fig.savefig(os.path.join(plot_dir, f'{block.info.blockname}.pdf'))
            plt.close(fig)
