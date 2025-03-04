


from fptools.measure import collect_signals
from fptools.viz import plot_cumulative_events, sig_catplot, plot_event_raster


def test_sig_catplot(tdt_preprocessed_sessions):
    for session in tdt_preprocessed_sessions:
        session.add_signal(collect_signals(session, 'RNP', 'Dopamine', start=-1, stop=3))

    sig_catplot(tdt_preprocessed_sessions,
                "Dopamine@RNP",
                col="paradigm_day",
                row="cube",
                hue="genotype")

def test_cumulative(ma_preprocessed_sessions):
    event_df = ma_preprocessed_sessions.epoc_dataframe(include_epocs=["rewarded_nosepoke"])
    plot_cumulative_events(event_df,
                           col="paradigm_day",
                           hue="genotype",
                           event="rewarded_nosepoke",
                           individual="mouseID")

def test_raster(ma_preprocessed_sessions):
    event_df = ma_preprocessed_sessions.epoc_dataframe(include_epocs=["rewarded_nosepoke"])
    plot_event_raster(event_df,
                      col="paradigm_day",
                      row="genotype",
                      event="rewarded_nosepoke",
                      individual="mouseID",
                      sort_col="D4")
