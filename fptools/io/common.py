from typing import Any, Literal, Optional, Protocol, TypedDict, cast


class SignalMapping(TypedDict):
    tdt_name: str
    dest_name: str
    role: Literal["experimental", "control"]


class Loader(Protocol):
    def __call__(self, path: str):
        pass


class DataTypeAdaptor:
    def __init__(self):
        self.name: str
        self.path: str
        self.loader: Loader
