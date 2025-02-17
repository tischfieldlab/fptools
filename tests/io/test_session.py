from pathlib import Path
from fptools.io.session import Session


def test_session_save_load(tdt_preprocessed_sessions, tmp_path: Path):
    for session in tdt_preprocessed_sessions:
        dest = tmp_path.joinpath(f'{session.name}.h5')
        session.save(dest)
        reconsitituted = Session.load(dest)

        assert session == reconsitituted
