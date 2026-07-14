import json
import os
import threading
from pathlib import Path
from typing import Dict

import util

file_lock = threading.Lock()


def ensure_directories(dir_map: Dict[str, str]):
    root = os.path.dirname(os.path.abspath(__file__))
    for dir_name in dir_map.values():
        filepath = os.path.join(root, dir_name)
        if not os.path.isdir(filepath):
            os.makedirs(filepath)


async def check_participant(mapping_file: str, pid: str) -> str | None:
    current_mappings = {}
    with file_lock:
        if os.path.exists(mapping_file) and os.path.getsize(mapping_file):
            with open(mapping_file, "r") as file:
                current_mappings = json.load(file)
    if pid in current_mappings:
        return current_mappings[pid]
    return None


async def create_participant_id(mapping_file: str, pid: str, randomise_id: bool) -> str:
    with file_lock:
        if os.path.exists(mapping_file) and os.path.getsize(mapping_file):
            with open(mapping_file, "r") as file:
                current_mappings = json.load(file)
        else:
            current_mappings = {}

        used = {v for v in current_mappings.values()}
        new_id = pid
        if randomise_id:
            new_id = util.generate_random_id(24)
            while new_id in used:
                new_id = util.generate_random_id(24)
        current_mappings[pid] = new_id

        with open(mapping_file, "w") as file:
            json.dump(current_mappings, file)
    return new_id


async def create_participant_file(directory: str, ppt_id: str):
    filename = os.path.join(directory, f"{ppt_id}.json")
    Path(filename).touch()


async def write_participant_file(directory: str, ppt_id: str, json_data: dict):
    filename = os.path.join(directory, f"{ppt_id}.json")
    with open(filename, "w") as outfile:
        json.dump(json_data, outfile)


async def move_participant_file(src_directory: str, dest_directory: str, ppt_id: str):
    filename = f"{ppt_id}.json"
    src_filename = os.path.join(src_directory, filename)
    dest_filename = os.path.join(dest_directory, filename)
    os.rename(src_filename, dest_filename)
