"""Run metadata (every job writes its configuration, git hash and environment next to its outputs) and retried
file-system operations."""

import json
import os
import platform
import socket
import subprocess
import sys
import time


def _git(*args):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.check_output(["git", "-C", root, *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def git_info():
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
        "dirty_files": None if not status else status.splitlines()[:50],
    }


IO_WAITS = (5, 10, 20, 40, 60, 60, 60)   # s between attempts: ~4.5 min in all


def retry_io(fn, what, waits=IO_WAITS):
    """fn() retried on OSError with the given waits: a transient file-system error (for example a shared file system
    that refuses writes for a few minutes) does not end the run. The last error is raised after the last wait."""
    for k in range(len(waits) + 1):
        try:
            return fn()
        except OSError as e:
            if k == len(waits):
                raise
            print(f"[io] {what} failed ({e!r}); retry {k + 1}/{len(waits)} in {waits[k]} s", flush=True)
            time.sleep(waits[k])


def append_line(path, line, waits=IO_WAITS):
    """Append one line to a log file, retrying a failed write (see retry_io)."""
    def write():
        with open(path, "a") as f:
            f.write(line + "\n")
    retry_io(write, f"append to {path}", waits)


def write_meta(out_dir, stage, config, extra=None):
    """Write <out_dir>/meta_<stage>.json and return the dict."""
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "stage": stage,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv,
        "config": config,
        "git": git_info(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if extra:
        meta.update(extra)
    with open(os.path.join(out_dir, f"meta_{stage}.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return meta
