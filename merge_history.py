#!/usr/bin/env python3
#
# This script merges ZSH history files together. It can be used for syncing history files from multiple separate hosts - eg cloud GPU instances - to a central server
# 
# On the client machine, run something like: `cat ~/.zsh_history | ssh target-host /path/to/merge_history.py HOSTTYPE` 
# Where HOSTTYPE is some identifier for this type of host, which is then used to separate out different ZSH histories.

import os
import sys
import time
import re
import fcntl
import argparse

LOCK_TIMEOUT = 20  # seconds
SLEEP_INTERVAL = 0.5  # seconds

parser = argparse.ArgumentParser(description='Merge zsh history from stdin into existing history file.')
parser.add_argument('typehost', default="runpod", nargs='?', type=str, help='Type of host to merge history for.')
args = parser.parse_args()

def read_zsh_history(file_obj):
    contents = file_obj.read()
    pattern = r": (\d+):(\d+);(.*?)(?=(?:: \d+:\d+;|$))"
    entries = re.findall(pattern, contents, re.DOTALL)
    return entries

def write_zsh_history(file_path, entries):
    with open(file_path, 'w') as f:
        for timestamp, duration, command in entries:
            f.write(f": {timestamp}:{duration};{command.rstrip()}\n")

def acquire_lock_or_timeout(file_obj):
    elapsed_time = 0
    while elapsed_time < LOCK_TIMEOUT:
        try:
            fcntl.flock(file_obj, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            time.sleep(SLEEP_INTERVAL)
            elapsed_time += SLEEP_INTERVAL
    return False

def main():
    os.makedirs(f"/workspace/{args.typehost}", exist_ok=True)
    existing_history_file = f"/workspace/{args.typehost}/.zsh_history"

    with open(existing_history_file, 'a') as f:
        if not acquire_lock_or_timeout(f):
            print("Error: Unable to acquire lock on .zsh_history within timeout.", file=sys.stderr)
            sys.exit(1)

        try:
            new_entries = read_zsh_history(sys.stdin)
            existing_entries = read_zsh_history(open(existing_history_file, 'r'))

            # Merge, sort by timestamp, and deduplicate
            combined = sorted(set(existing_entries + new_entries), key=lambda x: int(x[0]))

            write_zsh_history(existing_history_file, combined)
        except BrokenPipeError:
            sys.stderr.close()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

if __name__ == "__main__":
    main()
