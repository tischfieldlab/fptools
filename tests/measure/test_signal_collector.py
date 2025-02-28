


from pytest import approx
from fptools.measure import collect_signals


def test_collect_signals(tdt_preprocessed_sessions):

    for session in tdt_preprocessed_sessions:
        sig = collect_signals(session,
                            event="RNP_",
                            signal="Dopamine",
                            start=-1,
                            stop=3)
        assert sig.duration.total_seconds() == approx(4.0, rel=sig.fs)
