from typing import Literal, Optional, Union

from ..common import Pipeline, PairedSignalList, Processor, _flatten_paired_signals, _remap_paired_signals
from ..steps import Dff, Downsample, TrimSignals, DblExpFit, MotionCorrect, Rename


class DxpMotionDffPipeline(Pipeline):
    """Process using a double exponential fit for detrending, producing df/f values.

    Implemented as described in:
    Simpson et al. Neuron, 2024. https://doi.org/10.1016/j.neuron.2023.11.016

    Pipeline steps:
    1) Signals are optionally trimmed to the optical system start.
    2) Signals are fit with a double exponential to detrend
    3) Signals are motion corrected using a control signal
    4) Signals are converted to dF/F
    5) Signals are optionally downsampled factor `downsample`
    """

    def __init__(
        self,
        signals: PairedSignalList,
        rename_map: Optional[dict[Literal["signals", "epocs", "scalars"], dict[str, str]]] = None,
        trim_begin: Union[None, Literal["auto"], float, int] = "auto",
        trim_end: Union[None, float, int] = None,
        downsample: Optional[int] = 10,
        plot: bool = True,
        plot_dir: Optional[str] = None,
    ):
        """Initialize this pipeline.

        Args:
            signals: list of signal names to be processed
            rename_map: dictionary of signal, epoc, and scalar names to be renamed
            trim_begin: if not None, trim that amount of time (in seconds) from the beginning of the signal. If "auto", use the offset stored in `block.scalars.Fi1i.ts` for trimming
            trim_end: if not None, trim that amount of time (in seconds) from the end of the signal.
            downsample: if not `None`, downsample signal by `downsample` factor.
            plot: whether to plot the results of each step
            plot_dir: directory to save plots to
        """
        steps: list[Processor] = []

        # step to allow user to rename various things
        if rename_map is not None:
            steps.append(
                Rename(
                    signals=rename_map.get("signals", None), epocs=rename_map.get("epocs", None), scalars=rename_map.get("scalars", None)
                )
            )
            signals = _remap_paired_signals(
                signals, rename_map.get("signals", {})
            )  # remap the signals to the new names for the remaining steps

        # step to allow the user to trim the signals
        if trim_begin is not None or trim_end is not None:
            steps.append(TrimSignals(_flatten_paired_signals(signals), begin=trim_begin, end=trim_end))

        steps.extend(
            [
                DblExpFit(_flatten_paired_signals(signals), apply=True),
                MotionCorrect(signals),
                Dff([(s, f"{s}_dxpfit") for s in _flatten_paired_signals(signals)], center=False),
            ]
        )

        # step to downsample the signals
        if downsample is not None:
            steps.append(Downsample(_flatten_paired_signals(signals), window=downsample, factor=downsample))

        super().__init__(steps=steps, plot=plot, plot_dir=plot_dir)


# def dxp_motion_dff(
#     session: Session,
#     signal_map: list[SignalMapping],
#     show_steps: bool = True,
#     plot_dir: str = "",
#     trim_extent: Union[None, Literal["auto"], float, tuple[float, float]] = "auto",
# ):
#     """Preprocess using a double exponential fit for detrending, producing df/f values.

#     Args:
#         session: the session to populate.
#         signal_map: mapping of signals to perform
#         show_steps: if `True`, produce diagnostic plots of the preprocessing steps.
#         plot_dir: path where diagnostic plots of the preprocessing steps should be saved.
#         trim_extent: specification for trimming. None disables trimming, auto uses the offset stored in `block.scalars.Fi1i.ts`, a single float trims that amount of time (in seconds) from the beginning, a tuple of two floats specifies the amount of time (in seconds) from the beginning and end to trim, respectively.
#     """
#     try:
#         if show_steps:
#             fig, axs = plt.subplots(6, 1, figsize=(24, 6 * 6))
#             palette = sns.color_palette("colorblind", n_colors=len(signal_map))

#         signals: list[Signal] = []
#         for sm in signal_map:
#             signals.append(session.signals[sm["tdt_name"]].copy(new_name=sm["dest_name"]))

#         if show_steps:
#             for i, sig in enumerate(signals):
#                 axs[0].plot(sig.time, sig.signal, label=sig.name, c=palette[i])
#             axs[0].set_title("Raw signal")
#             axs[0].legend()

#         # trim raw signal start to when the optical system came online
#         if trim_extent is not None:
#             for sig in signals:
#                 if trim_extent == "auto":
#                     trim_args = {"begin": int(session.scalars["Fi1i"][0] * sig.fs)}
#                 elif isinstance(trim_extent, float):
#                     trim_args = {"begin": int(trim_extent * sig.fs)}
#                 elif len(trim_extent) == 2:
#                     trim_args = {"begin": int(trim_extent[0] * sig.fs), "end": int(trim_extent[1] * sig.fs)}

#                 sig.signal, sig.time = trim(sig.signal, sig.time, **trim_args)

#             if show_steps:
#                 for i, sig in enumerate(signals):
#                     axs[1].plot(sig.time, sig.signal, label=sig.name, c=palette[i])
#                 axs[1].set_title("Trimmed Raw signal")
#                 axs[1].legend()
#         else:
#             axs[1].set_title("Trimming Disabled")

#         # detrend using a double exponential fit
#         fits: list[np.ndarray] = []
#         for sig in signals:
#             detrended_sig, fit = detrend_double_exponential(sig.time, sig.signal)
#             fits.append(fit)

#             if show_steps:
#                 axs[2].plot(sig.time, sig.signal, label=sig.name, c=palette[i])
#                 axs[2].plot(sig.time, fit, label=f"{sig.name} dExp Fit", c=sns.desaturate(palette[i], 0.3))

#             sig.signal = detrended_sig

#         if show_steps:
#             axs[2].set_title("Double Exponential Fit")
#             axs[2].legend()

#             for i, sig in enumerate(signals):
#                 axs[3].plot(sig.time, detrended_sig, label=sig.name, c=palette[i])
#             axs[3].set_title("De-trended signals")
#             axs[3].legend()

#         # correct for motion artifacts
#         ctrl_idx = next((i for i, sm in enumerate(signal_map) if sm["role"] == "control"), None)
#         if ctrl_idx is None:
#             raise ValueError('To use motion correction, at least one signal must be marked with `role`="Control" in the `signal_map`!')
#         for sm, sig in zip(signal_map, signals):
#             if sm["role"] == "experimental":
#                 motion_corrected, est_motion = estimate_motion(sig.signal, signals[ctrl_idx].signal)

#                 if show_steps:
#                     axs[4].plot(sig.time, motion_corrected, label=sig.name, c=palette[i])
#                     axs[4].plot(sig.time, est_motion, label=f"{sig.name} Est. Motion", c=sns.desaturate(palette[i], 0.3))

#         if show_steps:
#             axs[4].set_title("Motion Correction")
#             axs[4].legend()

#         # calculate dF/F
#         for sig, fit in zip(signals, fits):
#             sig.signal = (sig.signal / fit) * 100
#             sig.units = "ΔF/F"

#         if show_steps:
#             for i, sig in enumerate(signals):
#                 axs[5].plot(sig.time, sig.signal, label=sig.name, c=palette[i])
#             axs[5].set_title("Normalized")
#             axs[5].legend()

#         # construct Signals and add to the Session
#         for sig in signals:
#             session.add_signal(sig)

#         return session
#     except:
#         raise
#     finally:
#         if show_steps:
#             fig.savefig(os.path.join(plot_dir, f"{session.name}.png"), dpi=600)
#             fig.savefig(os.path.join(plot_dir, f"{session.name}.pdf"))
#             plt.close(fig)
