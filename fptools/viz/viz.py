from matplotlib.ticker import Locator
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
from matplotlib.axes import Axes
import pandas as pd
import scipy
import scipy.interpolate
import seaborn as sns

from fptools.io import Session, Signal
from fptools.preprocess.lib import are_arrays_same_length, fs2t










def aggregate_signals(signals: list[Signal], method='mean') -> Signal:
    """Aggregate the signals
    """
    if len(signals) <= 0:
        raise ValueError('No signals were passed!')

    # check all signals have the same number of samples
    assert np.all(np.equal([s.nsamples for s in signals], signals[0].nsamples))

    if method is not None:
        signals = [s.aggregate(method) for s in signals]

    s = Signal(signals[0].name, np.vstack([s.signal for s in signals]), time=signals[0].time, units=signals[0].units)
    s.marks.update(signals[0].marks)
    return s


def plot_signal(signal: Signal, ax: Axes = None, indv: bool = True, color: ColorType = 'k', indv_c: ColorType = 'b') -> mpl.axes.Axes:
    if ax is None:
        fig, ax = plt.subplots()

    df = pd.DataFrame(signal.signal.T)
    df.index = signal.time
    df = df.melt(ignore_index=False)

    if indv and signal.nobs:
        for i in range(signal.signal.shape[0]):
            sns.lineplot(data=None, x=signal.time, y=signal.signal[i, :], alpha=0.1, ax=ax, color=indv_c)

    sns.lineplot(data=df, x=df.index, y='value', ax=ax, color=color)

    ax.set_xlabel('Time, Reletive to Event (s)')
    ax.set_ylabel(f'{signal.name} ({signal.units})')

    xticks = ax.get_xticks()
    xticklabels = ax.get_xticklabels()
    for k, v in signal.marks.items():
        ax.axvline(v, c='gray', ls='--')
        try:
            xt = np.where(xticks == float(v))[0][0]
            xticklabels[xt] = k
        except:
            xticks = np.append(xticks, float(v))
            xticklabels.append(k)
            order = np.argsort(xticks)
            xticks = xticks[order]
            xticklabels = [xticklabels[i] for i in order]
            pass
    ax.set_xticks(xticks, xticklabels)

    return ax



def plot_heatmap(signal: Signal, ax=None, cmap="viridis", vmin=None, vmax=None):

    if ax is None:
        fig, ax = plt.subplots()

    cbar_kwargs = {
        'label': f'{signal.name} ({signal.units})'
    }

    sns.heatmap(data=np.atleast_2d(signal.signal),
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar_kws=cbar_kwargs)

    xticks = [0, signal.nsamples]
    xticklabels = [f'{signal.time[0]:0.0f}', f'{signal.time[-1]:0.0f}']
    for m, t in signal.marks.items():
        i = signal.tindex(t)
        ax.axvline(i, c='w', ls='--')
        xticks.append(i)
        xticklabels.append(f'{m}')
    order = np.argsort(xticks)
    xticks = [xticks[i] for i in order]
    xticklabels = [xticklabels[i] for i in order]
    ax.set_xticks(xticks, labels=xticklabels, rotation=0)

    ax.set_xlabel('Time, Reletive to Event (sec)')

    return ax



# def plot_signal_at_events(events, signal, time, ax=None, pre=1.0, post=2.0, indv=True, color='k', indv_c='b'):
#     new_time, accum = collect_signals(events, signal, time, pre=pre, post=post)

#     df = pd.DataFrame(accum.T)
#     df.index = new_time
#     df = df.melt(ignore_index=False)

#     if ax is None:
#         fig, ax = plt.subplots()

#     if indv:
#         for i in range(accum.shape[0]):
#             sns.lineplot(data=None, x=new_time, y=accum[i, :], alpha=0.1, ax=ax, color=indv_c)

#     sns.lineplot(data=df, x=df.index, y='value', ax=ax, color=color)
#     ax.axvline(0, c='k', ls='--')
#     ax.set_xlabel('Time, Reletive to Event (s)')

#     return accum, new_time



