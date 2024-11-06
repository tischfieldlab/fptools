import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fptools.io import Session, Signal, SignalMapping
from fptools.preprocess.lib import detrend_double_exponential, estimate_motion, trim


def dxp_motion_dff(session: Session, block: Any, signal_map: list[SignalMapping], show_steps=True, plot_dir=''):
    try:
        if show_steps:
            fig, axs = plt.subplots(6, 1, figsize=(24, 6*6))
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
            sig.signal, sig.time = trim(sig.signal, sig.time, begin=int(block.scalars.Fi1i.ts[0] * sig.fs))

        if show_steps:
            for i, sig in enumerate(signals):
                axs[1].plot(sig.time, sig.signal, label=sig.name, c=palette[i])
            axs[1].set_title('Trimmed Raw signal')
            axs[1].legend()

        # detrend using a double exponential fit
        fits: list[np.ndarray] = []
        for sig in signals:
            detrended_sig, fit = detrend_double_exponential(sig.time, sig.signal)
            fits.append(fit)

            if show_steps:
                axs[2].plot(sig.time, sig.signal, label=sig.name, c=palette[i])
                axs[2].plot(sig.time, fit, label=f'{sig.name} dExp Fit', c=sns.desaturate(palette[i], 0.3))

            sig.signal = detrended_sig

        if show_steps:
            axs[2].set_title('Double Exponential Fit')
            axs[2].legend()

            for i, sig in enumerate(signals):
                axs[3].plot(sig.time, detrended_sig, label=sig.name, c=palette[i])
            axs[3].set_title('De-trended signals')
            axs[3].legend()

        # correct for motion artifacts
        ctrl_idx = next((i for i, sm in enumerate(signal_map) if sm['role'] == "control"), None)
        if ctrl_idx is None:
            raise ValueError('To use motion correction, at least one signal must be marked with `role`="Control" in the `signal_map`!')
        for sm, sig in zip(signal_map, signals):
            if sm['role'] == 'experimental':
                motion_corrected, est_motion = estimate_motion(sig.signal, signals[ctrl_idx].signal)

                if show_steps:
                    axs[4].plot(sig.time, motion_corrected, label=sig.name, c=palette[i])
                    axs[4].plot(sig.time, est_motion, label=f'{sig.name} Est. Motion', c=sns.desaturate(palette[i], 0.3))

        if show_steps:
            axs[4].set_title('Motion Correction')
            axs[4].legend()

        # calculate dF/F
        for sig, fit in zip(signals, fits):
            sig.signal = (sig.signal / fit) * 100
            sig.units = 'ΔF/F'

        if show_steps:
            for i, sig in enumerate(signals):
                axs[5].plot(sig.time, sig.signal, label=sig.name, c=palette[i])
            axs[5].set_title('Normalized')
            axs[5].legend()

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
