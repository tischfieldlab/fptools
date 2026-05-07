import glob
import os
from typing import Optional
from pathlib import Path
import datetime
import re

import pandas as pd
import numpy as np

from .common import DataTypeAdaptor, Loader, DataLocator
from .session import Session, Signal
from .tdt_with_dlc import DLCLoader

def find_ma_blocks(path: str, pattern: str = "*.txt") -> list[DataTypeAdaptor]:
    """Data Locator for med-associates files.

    Given a path to a directory, will search that path recursively for med-associates files.

    Args:
        path: path to search for med-associates data files

    Returns:
        list of DataTypeAdaptor, each adaptor corresponding to one session, of data to be loaded
    """
    found_files = list(glob.glob(os.path.join(path, "**", pattern), recursive=True))
    filtered = list(filter(is_file_ma, found_files))
    items_out = []
    for f in filtered:
        adapt = DataTypeAdaptor()
        adapt.path = f  # path to the med-associates txt file
        adapt.name = os.path.splitext(os.path.basename(adapt.path))[0]  # the basename of the file without file extensions
        adapt.loaders.append(parse_ma_session)
        items_out.append(adapt)
    return items_out

class FindMADLCBlocks(DataLocator):

    def __init__(self, model_name: Optional[list[str]] = None, filtered_only: bool = True):
        """Initialize this MA-DLC Data Locator.

        Args:
            model_name: If provided, only look for DLC files with that model name(s), If None, load all files that look like DLC data
            filtered_only: If true, only load filtered DLC data, otherwise, load any DLC data
        """
        self.model_name = model_name
        self.filtered_only = filtered_only

    def __call__(self, path: str) -> list[DataTypeAdaptor]:
        """Data Locator for MA blocks with DLC data.

        Given a path to a directory, will search that path recursively for MA blocks that contain DLC output files in .h5 format.

        Args:
            path: path to search for MA blocks with DLC data

        Returns:
            list of DataTypeAdaptor, each adaptor corresponding to one session, of data to be loaded
        """
        found_files = list(glob.glob(os.path.join(path, "**", "*.txt"), recursive=True))
        filtered = list(filter(is_file_ma, found_files))

        items_out = []
        for file in filtered:
            adapt = DataTypeAdaptor()
            adapt.path = os.path.dirname(file)  # the directory for the block
            adapt.name = os.path.basename(adapt.path)  # the name of the block folder
            adapt.loaders.append(MALoader())
            adapt.loaders.append(DLCLoader(model_name=self.model_name, filtered_only=self.filtered_only))
            items_out.append(adapt)

        return items_out

def is_file_ma(path: str) -> bool:
    """Test if a file looks like a med-associates data file.

    Args:
        path: path to a file to test.

    Returns:
        True if the file looks like a med-associates file, otherwise False.
    """
    with open(path, mode="r") as f:
        first_line = f.readline()
        if first_line.startswith("File: "):
            return True
    return False


_rx_dict: dict[str, re.Pattern] = {
    "StartDate": re.compile(r"^Start Date: (?P<StartDate>.*)\r\n"),
    "EndDate": re.compile(r"^End Date: (?P<EndDate>.*)\r\n"),
    "StartTime": re.compile(r"^Start Time: (?P<StartTime>.*)\r\n"),
    "EndTime": re.compile(r"^End Time: (?P<EndTime>.*)\r\n"),
    "Subject": re.compile(r"^Subject: (?P<Subject>.*)\r\n"),
    "Experiment": re.compile(r"^Experiment: (?P<Experiment>.*)\r\n"),
    "Group": re.compile(r"^Group: (?P<Group>.*)\r\n"),
    "Box": re.compile(r"^Box: (?P<Box>.*)\r\n"),
    "MSN": re.compile(r"^MSN: (?P<MSN>.*)\r\n"),
    "SCALAR": re.compile(r"(?P<name>[A-Z]{1}): *(?P<value>\d+\.\d*)\r\n"),
    "ARRAY": re.compile(r"(?P<name>[A-Z]{1}):\r\n"),
    "ARRAYidx": re.compile(r"^ *(?P<index>[0-9]+):(?P<list>.*)\r\n"),
    "STARTOFDATA": re.compile(r"\r\r\n"),
}


def _parse_line(line: str):
    """Parse a single session data file line.

    Do a regex search against all defined regexes and
    return the key and match result of the first matching regex

    Args:
        line: the line to parse

    Returns:
        (None, None) if no match was found, otherwise the key and match object
    """
    for key, rx in _rx_dict.items():
        match = rx.search(line)
        if match:
            return key, match
    # if there are no matches
    return None, None

class MALoader:
    
    # def __init__(
    #     self,
    #     exclude_epocs: Optional[list[str]] = None,
    #     exclude_scalars: Optional[list[str]] = None,
    # ) -> None:
    #     """Initialize this MALoader."""
    #     self.exclude_epocs = exclude_epocs or []
    #     self.exclude_scalars = exclude_scalars or []
    
    def __call__(self, session: Session, path: str) -> Session:
        """Data Loader for MA blocks.

        Args:
            session: the session for data to be loaded into
            path: path to a MA block folder

        Returns:
            Session object with data added
        """
        MPCDateStringRe = re.compile(r"\s*(?P<hour>[0-9]+):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})")
        
        # find the med associates .txt file
        found_files = list(glob.glob(os.path.join(path, "**", "*.txt"), recursive=True))
        filtered = list(filter(is_file_ma, found_files))

        # open the file and read through it line by line
        with open(filtered[0], "r", newline="\n") as file_object:

            session.metadata["blockname"] = os.path.basename(path)

            line = file_object.readline()
            while line:
                # at each line check for a match with a regex
                key, match = _parse_line(line)

                # start of data is '\r\r\n'
                if key == "STARTOFDATA":
                    pass

                # extract start date
                if key == "StartDate":
                    session.metadata["StartDate"] = datetime.datetime.strptime(match.group(key), "%m/%d/%y").date()

                # extract end date
                if key == "EndDate":
                    session.metadata["EndDate"] = datetime.datetime.strptime(match.group(key), "%m/%d/%y").date()

                # extract start time
                if key == "StartTime":
                    date_match = MPCDateStringRe.search(match.group(key))
                    if date_match is not None:
                        (hour, min, sec) = [int(date_match.group(g)) for g in ["hour", "minute", "second"]]
                        session.metadata["StartTime"] = datetime.time(hour, min, sec)
                        # date should be already read
                        session.metadata["StartDateTime"] = datetime.datetime.combine(
                            session.metadata["StartDate"], session.metadata["StartTime"]
                        )

                # extract end time
                if key == "EndTime":
                    date_match = MPCDateStringRe.search(match.group(key))
                    if date_match is not None:
                        (hour, min, sec) = [int(date_match.group(g)) for g in ["hour", "minute", "second"]]
                        session.metadata["EndTime"] = datetime.time(hour, min, sec)
                        # date should be already read
                        session.metadata["EndDateTime"] = datetime.datetime.combine(session.metadata["EndDate"], session.metadata["EndTime"])

                # extract Subject
                if key == "Subject":
                    session.metadata["Subject"] = match.group(key)

                # extract Experiment
                if key == "Experiment":
                    session.metadata["Experiment"] = match.group(key)

                # extract Group
                if key == "Group":
                    session.metadata["Group"] = match.group(key)

                # extract Box
                if key == "Box":
                    session.metadata["Box"] = int(match.group(key))

                # extract MSN
                if key == "MSN":
                    session.metadata["MSN"] = match.group(key)

                # extract scalars
                if key == "SCALAR":
                    session.scalars[match.group("name")] = np.array([float(match.group("value"))])

                # identify an array
                if key == "ARRAY":
                    # print(f'This is the beginning of an Array:: "{line}"')
                    # have now have to step through the array
                    file_tell = file_object.tell()
                    subline = file_object.readline()
                    # print(f'This is the first line of the array:: "{subline}"')
                    items = []
                    while subline:
                        m = _rx_dict["ARRAYidx"].search(subline)
                        if m is not None:
                            items.extend([float(l) for l in m.group("list").split()])

                        else:
                            # have to rewind
                            # print(f'This is one line beyond the last line of the array:: "{subline}"')
                            file_object.seek(file_tell)
                            break
                        file_tell = file_object.tell()
                        subline = file_object.readline()
                    # print(f'Setting "{match.group("name")}"={items}')
                    session.epocs[match.group("name")] = np.array(items)
                line = file_object.readline()
                
        return session

def parse_ma_session(session: Session, path: str) -> Session:
    """Parse a session data file from MedAssociates.

    Adapted from https://github.com/matthewperkins/MPCdata, but fixes some issues.

    Args:
        path: Filepath for file_object to be parsed

    Returns:
        SessionCollection
    """
    # print(path)
    MPCDateStringRe = re.compile(r"\s*(?P<hour>[0-9]+):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})")
    # open the file and read through it line by line
    with open(path, "r", newline="\n") as file_object:
        line = file_object.readline()
        while line:
            # at each line check for a match with a regex
            key, match = _parse_line(line)

            # start of data is '\r\r\n'
            if key == "STARTOFDATA":
                pass

            # extract start date
            if key == "StartDate":
                session.metadata["StartDate"] = datetime.datetime.strptime(match.group(key), "%m/%d/%y").date()

            # extract end date
            if key == "EndDate":
                session.metadata["EndDate"] = datetime.datetime.strptime(match.group(key), "%m/%d/%y").date()

            # extract start time
            if key == "StartTime":
                date_match = MPCDateStringRe.search(match.group(key))
                if date_match is not None:
                    (hour, min, sec) = [int(date_match.group(g)) for g in ["hour", "minute", "second"]]
                    session.metadata["StartTime"] = datetime.time(hour, min, sec)
                    # date should be already read
                    session.metadata["StartDateTime"] = datetime.datetime.combine(
                        session.metadata["StartDate"], session.metadata["StartTime"]
                    )

            # extract end time
            if key == "EndTime":
                date_match = MPCDateStringRe.search(match.group(key))
                if date_match is not None:
                    (hour, min, sec) = [int(date_match.group(g)) for g in ["hour", "minute", "second"]]
                    session.metadata["EndTime"] = datetime.time(hour, min, sec)
                    # date should be already read
                    session.metadata["EndDateTime"] = datetime.datetime.combine(session.metadata["EndDate"], session.metadata["EndTime"])

            # extract Subject
            if key == "Subject":
                session.metadata["Subject"] = match.group(key)

            # extract Experiment
            if key == "Experiment":
                session.metadata["Experiment"] = match.group(key)

            # extract Group
            if key == "Group":
                session.metadata["Group"] = match.group(key)

            # extract Box
            if key == "Box":
                session.metadata["Box"] = int(match.group(key))

            # extract MSN
            if key == "MSN":
                session.metadata["MSN"] = match.group(key)

            # extract scalars
            if key == "SCALAR":
                session.scalars[match.group("name")] = np.array([float(match.group("value"))])

            # identify an array
            if key == "ARRAY":
                # print(f'This is the beginning of an Array:: "{line}"')
                # have now have to step through the array
                file_tell = file_object.tell()
                subline = file_object.readline()
                # print(f'This is the first line of the array:: "{subline}"')
                items = []
                while subline:
                    m = _rx_dict["ARRAYidx"].search(subline)
                    if m is not None:
                        items.extend([float(l) for l in m.group("list").split()])

                    else:
                        # have to rewind
                        # print(f'This is one line beyond the last line of the array:: "{subline}"')
                        file_object.seek(file_tell)
                        break
                    file_tell = file_object.tell()
                    subline = file_object.readline()
                # print(f'Setting "{match.group("name")}"={items}')
                session.epocs[match.group("name")] = np.array(items)
            line = file_object.readline()
    return session