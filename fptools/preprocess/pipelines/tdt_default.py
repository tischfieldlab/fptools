from typing import Any, Literal, Optional, Union

from ..steps import Downsample, Rename, TrimSignals, MotionCorrect, Dff
from ..common import Pipeline, PairedSignalList, Processor, _flatten_paired_signals, _remap_paired_signals


class TdtDefaultPipeline(Pipeline):
    """Process using a the pipeline described by TDT.

    Implemented as described in:
    https://www.tdt.com/docs/sdk/offline-data-analysis/offline-data-python/examples/

    Pipeline steps:
    1) Signals are optionally trimmed.
    2) Signals are fit with a linear model against a control signal
    3) Signals are then detrended and dF/F calculated
    4) Signals are optionally downsampled factor `downsample`
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

        steps.append(MotionCorrect(signals))
        steps.append(Dff([(s, f"{s}_motion_est") for s, _ in signals], center=False))

        if downsample is not None:
            steps.append(Downsample(_flatten_paired_signals(signals), window=downsample, factor=downsample))

        super().__init__(steps=steps, plot=plot, plot_dir=plot_dir)


# def tdt_default(
#     session: Session,
#     signal_map: list[SignalMapping],
#     show_steps: bool = True,
#     plot_dir: str = "",
#     trim_extent: Union[None, Literal["auto"], float, tuple[float, float]] = "auto",
#     downsample: Optional[int] = None,
# ) -> Session:
#     """A preprocess pipeline based on TDT tutorials.

#     Pipeline steps:
#     1) Signals are trimmed to the optical system start.
#     2) Perform linear regression between sensor and isosbestic
#     3) calculate dFF
#     #4) Signals are optionally downsampled factor `downsample`

#     Args:
#         session: the session to populate.
#         signal_map: mapping of signals to perform
#         show_steps: if `True`, produce diagnostic plots of the preprocessing steps.
#         plot_dir: path where diagnostic plots of the preprocessing steps should be saved.
#         trim_extent: specification for trimming. None disables trimming, auto uses the offset stored in `block.scalars.Fi1i.ts`, a single float trims that amount of time (in seconds) from the beginning, a tuple of two floats specifies the amount of time (in seconds) from the beginning and end to trim, respectively.
#         downsample: if not `None`, downsample signal by `downsample` factor.
#     """
#     try:
#         if show_steps:
#             fig, axs = plt.subplots(4, 1, figsize=(24, 6 * 4))
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

#         ctrl_idx = next((i for i, sm in enumerate(signal_map) if sm["role"] == "control"), None)
#         if ctrl_idx is None:
#             raise ValueError('at least one signal must be marked with `role`="Control" in the `signal_map`!')
#         for sm, sig in zip(signal_map, signals):
#             if sm["role"] == "experimental":

#                 x = np.array(signals[ctrl_idx].signal)
#                 y = np.array(sig.signal)
#                 bls = np.polyfit(x, y, 1)
#                 Y_fit_all = np.multiply(bls[0], x) + bls[1]
#                 Y_dF_all = y - Y_fit_all
#                 dFF = np.multiply(100, np.divide(Y_dF_all, Y_fit_all))
#                 sig.signal = dFF
#                 sig.units = "ΔF/F"

#         if show_steps:
#             for i, sig in enumerate(signals):
#                 axs[2].plot(sig.time, sig.signal, label=sig.name, c=palette[i])
#             axs[2].set_title("Normalized (dff)")
#             axs[2].legend()

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
