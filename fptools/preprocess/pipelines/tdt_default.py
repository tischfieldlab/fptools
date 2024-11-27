import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fptools.preprocess.lib import lowpass_filter, t2fs, trim, fs2t, downsample as downsample_fn
from fptools.io import Session, Signal, SignalMapping


def tdt_default(
    session: Session,
    block: Any,
    signal_map: list[SignalMapping],
    show_steps: bool = True,
    plot_dir: str = "",
    downsample: Optional[int] = None,
) -> Session:
    """A preprocess pipeline based on TDT tutorials.

    Pipeline steps:
    1) Signals are trimmed to the optical system start.
    2) Perform linear regression between sensor and isosbestic
    3) calculate dFF
    #4) Signals are optionally downsampled factor `downsample`

    Args:
        session: the session to populate.
        block: block data struct from `tdt.read_block()`.
        signal_map: mapping of signals to perform
        show_steps: if `True`, produce diagnostic plots of the preprocessing steps.
        plot_dir: path where diagnostic plots of the preprocessing steps should be saved.
        downsample: if not `None`, downsample signal by `downsample` factor.
    """
    try:
        if show_steps:
            fig, axs = plt.subplots(4, 1, figsize=(24, 6 * 4))
            palette = sns.color_palette("colorblind", n_colors=len(signal_map))

        signals: list[Signal] = []
        for sm in signal_map:
            stream = block.streams[sm["tdt_name"]]
            signals.append(Signal(sm["dest_name"], stream.data, fs=stream.fs))

        if show_steps:
            for i, sig in enumerate(signals):
                axs[0].plot(sig.time, sig.signal, label=sig.name, c=palette[i])
            axs[0].set_title("Raw signal")
            axs[0].legend()

        # trim raw signal start to when the optical system came online
        for sig in signals:
            sig.signal, sig.time = trim(sig.signal, sig.time, begin=int(block.scalars.Fi1i.ts[0] * sig.fs))

        if show_steps:
            for i, sig in enumerate(signals):
                axs[1].plot(sig.time, sig.signal, label=sig.name, c=palette[i])
            axs[1].set_title("Trimmed Raw signal")
            axs[1].legend()


        ctrl_idx = next((i for i, sm in enumerate(signal_map) if sm["role"] == "control"), None)
        if ctrl_idx is None:
            raise ValueError('at least one signal must be marked with `role`="Control" in the `signal_map`!')
        for sm, sig in zip(signal_map, signals):
            if sm["role"] == "experimental":

                x = np.array(signals[ctrl_idx].signal)
                y = np.array(sig.signal)
                bls = np.polyfit(x, y, 1)
                Y_fit_all = np.multiply(bls[0], x) + bls[1]
                Y_dF_all = y - Y_fit_all
                dFF = np.multiply(100, np.divide(Y_dF_all, Y_fit_all))
                sig.signal = dFF
                sig.units = "ΔF/F"

        if show_steps:
            for i, sig in enumerate(signals):
                axs[2].plot(sig.time, sig.signal, label=sig.name, c=palette[i])
            axs[2].set_title("Normalized (dff)")
            axs[2].legend()

        # construct Signals and add to the Session
        for sig in signals:
            session.add_signal(sig)

        return session
    except:
        raise
    finally:
        if show_steps:
            fig.savefig(os.path.join(plot_dir, f"{block.info.blockname}.png"), dpi=600)
            fig.savefig(os.path.join(plot_dir, f"{block.info.blockname}.pdf"))
            plt.close(fig)
