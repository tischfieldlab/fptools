from typing import Literal, Protocol, TypedDict

from .session import Session


class SignalMapping(TypedDict):
    tdt_name: str
    dest_name: str
    role: Literal["experimental", "control"]


class SignalMapping2(TypedDict):
    experimental: str
    control: str


class DataTypeAdaptor:
    def __init__(self) -> None:
        """Initialize this DataTypeAdaptor."""
        self.name: str
        self.path: str
        self.loaders: list[Loader] = []


class DataLocator(Protocol):
    def __call__(self, path: str) -> list[DataTypeAdaptor]:
        """Data Locator Protocol.

        A data locator should take a path, and search that path for data files
        to be loaded into a session. For each candidate, it should generate a DataTypeAdaptor

        Args:
            path: path to search for data files

        Returns:
            list of DataTypeAdaptor, each adaptor corresponding to one session, of data to be loaded
        """
        pass


class Loader(Protocol):
    def __call__(self, session: Session, path: str) -> Session:
        """Data Loader Protocol.

        A Loader receives a session instance and a path to data. It should load this data into
        the Session instance and return back the session.

        Args:
            session: the session for data to be loaded into
            path: path to some data that should be loaded

        Returns:
            Session object with data added
        """
        pass


class Preprocessor(Protocol):
    def __call__(self, session: Session, signal_map: list[SignalMapping], **kwargs) -> Session:
        """Preprocessor protocol.

        Args:
            session: the session to operate upon
            signal_map: mapping of signal information
            **kwargs: additional kwargs a preprocessor might need

        Returns:
            Session object with preprocessed data added.
        """
        pass
