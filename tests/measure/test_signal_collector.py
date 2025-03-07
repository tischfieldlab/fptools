


from pytest import approx
from fptools.measure import collect_signals, collect_signals_2event


def test_collect_signals(tdt_preprocessed_sessions):

    for session in tdt_preprocessed_sessions:
        sig = collect_signals(session,
                            event="RNP",
                            signal="Dopamine",
                            start=-1,
                            stop=3)
        assert sig.duration.total_seconds() == approx(4.0, rel=sig.fs)


def test_collect_signals_2event(tdt_preprocessed_sessions):

    for session in tdt_preprocessed_sessions:
        sig = collect_signals_2event(session,
                            event1="RNP",
                            event2="RMG",
                            signal="Dopamine",
                            pre=2.0,
                            inter=2.0,
                            post=2.0)
        assert sig.duration.total_seconds() == approx(6.0, rel=sig.fs)
