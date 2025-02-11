


from fptools.measure import measure_peaks


def test_measure_peaks(tdt_preprocessed_sessions):
    measure_peaks(tdt_preprocessed_sessions, signal="Dopamine")
