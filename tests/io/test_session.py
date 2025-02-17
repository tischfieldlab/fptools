from pathlib import Path
from fptools.io.session import save_session, load_session


def test_session_save_load(tdt_preprocessed_sessions, tmp_path: Path):
    for session in tdt_preprocessed_sessions:
        dest = tmp_path.joinpath(f'{session.name}.h5')
        save_session(session, dest)
        reconsitituted = load_session(dest)

        assert session == reconsitituted
