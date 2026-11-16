"""Turn a run's offline W&B log into train_log.json (Sec. 4.1, App. D, E and G).

The training code logs the replay-batch critic value `training/q` and the latest practice outcome
`is_success (exploration)` every `log_interval` gradient steps. This module keeps those two values
per logged step. The critic values give the replay-batch Q of Sec. 4.1 and App. E. The practice
outcomes give S_W on the approximate rerun (App. D) and its window choice (App. G).

The .wandb file is a LevelDB-style log: a 7-byte file header that starts with ":W&B", then 32 KiB
blocks of chunks. Each chunk has a 7-byte header (4-byte CRC, 2-byte little-endian length, 1-byte
type). A record is one FULL chunk, or a FIRST chunk, any MIDDLE chunks and a LAST chunk. Each record
is one protobuf `Record` of wandb's internal schema. The reader is written out here because wandb
no longer ships a Python reader for these files. CRCs are not checked.

    python -m ckpt_selection.training_log --run TAG [--outputs DIR]
"""

import argparse
import json
import struct
import warnings
from pathlib import Path

from ckpt_selection import config as config_lib
from ckpt_selection import layout

FILE_MAGIC = b":W&B"  # first bytes of every .wandb file
FILE_HEADER_BYTES = 7  # magic (4), format id (2), version (1)
BLOCK_BYTES = 32 * 1024  # LevelDB log block size
CHUNK_HEADER_FORMAT = "<IHB"  # CRC (uint32), payload length (uint16), chunk type (uint8)
CHUNK_HEADER_BYTES = struct.calcsize(CHUNK_HEADER_FORMAT)
CHUNK_PADDING = 0
CHUNK_FULL = 1
CHUNK_FIRST = 2
CHUNK_MIDDLE = 3
CHUNK_LAST = 4

STEP_KEY = "_step"
CRITIC_KEY = "training/q"  # replay-batch Q, the mean over the critic ensemble (App. E)
PRACTICE_KEY = "is_success (exploration)"  # outcome of the latest practice episode (App. G)
KEPT_KEYS = (PRACTICE_KEY, CRITIC_KEY)


def read_chunks(data: bytes):
    """Yield (chunk type, payload) for every chunk after the file header."""
    if data[: len(FILE_MAGIC)] != FILE_MAGIC:
        raise ValueError("not a W&B log: the file does not start with ':W&B'")
    position = FILE_HEADER_BYTES
    while position + CHUNK_HEADER_BYTES <= len(data):
        block_offset = position % BLOCK_BYTES
        if BLOCK_BYTES - block_offset < CHUNK_HEADER_BYTES:
            position += BLOCK_BYTES - block_offset  # block trailer too short for a header
            continue
        _crc, length, chunk_type = struct.unpack_from(CHUNK_HEADER_FORMAT, data, position)
        if chunk_type == CHUNK_PADDING:
            position += BLOCK_BYTES - block_offset  # zero padding up to the end of the block
            continue
        position += CHUNK_HEADER_BYTES
        yield chunk_type, data[position : position + length]
        position += length


def read_messages(path: Path):
    """Yield every protobuf Record in a .wandb file."""
    # Imported here so that load.py and selectors.py, which use only this module's key names,
    # run without wandb installed.
    from wandb.proto import wandb_internal_pb2

    with open(path, "rb") as handle:
        data = handle.read()
    pending = b""
    for chunk_type, payload in read_chunks(data):
        if chunk_type == CHUNK_FULL:
            whole = payload
        elif chunk_type == CHUNK_FIRST:
            pending = payload
            continue
        elif chunk_type == CHUNK_MIDDLE:
            pending += payload
            continue
        elif chunk_type == CHUNK_LAST:
            whole = pending + payload
        else:
            raise ValueError(f"{path}: unknown chunk type {chunk_type}")
        pending = b""
        record = wandb_internal_pb2.Record()
        record.ParseFromString(whole)
        yield record
    if pending:
        warnings.warn(f"{path}: the file ends inside a split record; its last record is dropped")


def parse_training_log(path: Path) -> list[dict]:
    """Return one row per logged step that has a critic value, sorted by step."""
    by_step = {}
    for record in read_messages(path):
        if record.WhichOneof("record_type") != "history":
            continue
        values = {}
        for item in record.history.item:
            # Newer writers leave `key` empty and split "training/q" into nested_key parts.
            key = item.key or "/".join(item.nested_key)
            values[key] = json.loads(item.value_json)
        if STEP_KEY not in values:
            continue
        row = by_step.setdefault(int(values[STEP_KEY]), {})
        row.update({key: values[key] for key in KEPT_KEYS if key in values})
    rows = []
    for step in sorted(by_step):
        if CRITIC_KEY in by_step[step]:
            rows.append({STEP_KEY: step, **by_step[step]})
    return rows


def main() -> None:
    """Parse one run's offline W&B log and write its train_log.json."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run tag from config/experiments.toml")
    parser.add_argument("--outputs", default=None, help="outputs folder (default: from config)")
    args = parser.parse_args()

    config = config_lib.load_config()
    if "train_log" not in config_lib.run_entry(config, args.run)["stages"]:
        raise ValueError(f"run {args.run}: its stages in the config do not list train_log")
    outputs = config_lib.outputs_root(config, args.outputs)
    source = layout.wandb_file(outputs, args.run)
    rows = parse_training_log(source)
    if not rows:
        raise ValueError(f"{source}: no logged step has {CRITIC_KEY!r}")
    destination = layout.train_log_path(outputs, args.run)
    layout.write_json(destination, rows)
    print(f"{args.run}: {len(rows)} logged steps, {rows[0][STEP_KEY]}..{rows[-1][STEP_KEY]}")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
