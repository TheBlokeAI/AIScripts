#
# Download models from Hugging Face Hub, including at fast speed using hf_transfer
# NOTE: requires `pip3 install hf_transfer` for fast transfers (the default)
#

#TODO: disable fast downloads if hf_transfer is not installed

import logging
import time
import argparse
import os
import sys
from multiprocessing import Process, Queue
import threading

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

def total_size(source):
    size = os.path.getsize(source)
    for item in os.listdir(source):
        itempath = os.path.join(source, item)
        if os.path.isfile(itempath):
            size += os.path.getsize(itempath)
        elif os.path.isdir(itempath):
            size += total_size(itempath)
    return size

def get_size(model_folder, repo_id, symlinks="auto"):
    if symlinks.lower() == 'auto':
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == repo_id:
                model_folder = repo.repo_path
    size = total_size(model_folder)
    size_MB = size / (1024 ** 2)
    size_GB = size / (1024 ** 3)
    return size, size_MB, size_GB

class RepeatTimer(threading.Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)

def log_size(start_time, model_folder, repo_id, symlinks="auto"):
    size, size_MB, size_GB = get_size(model_folder, repo_id, symlinks=symlinks)
    seconds = time.time() - start_time
    logger.info(f'Elapsed time: {seconds:.2f} seconds. Downloaded {size} bytes ({size_GB:.2f} GB) so far at a rate of: {size_MB / seconds:.2f} MB/s')

def run_snapshot_download(repo_id, local_dir, queue, branch="main", token=True, ignore_patterns=[], local_dir_use_symlinks="auto"):
    from huggingface_hub import snapshot_download
    retry = True
    try_count = 0
    max_tries = 5
    while retry and try_count < max_tries:
        try:
            snapshot_download(token=token,
                              repo_id=repo_id,
                              local_dir=local_dir,
                              revision=branch,
                              ignore_patterns=ignore_patterns,
                              local_dir_use_symlinks=local_dir_use_symlinks)
            retry = False
        except Exception as e:
            logger.error(f"Exception: {e}")
            try_count += 1
            logger.info(f"Retrying {try_count} of {max_tries}")
            time.sleep(1)
    if retry:
        logger.error(f"Failed to download {repo_id} after {max_tries} tries. Exiting.")
        queue.put(False)
    else:
        queue.put(True)

def hf_snapshot_download(repo_id, local_dir, branch="main", log_period=15, fast=True, cache_dir=None, ignore_patterns=[], local_dir_use_symlinks=False, token=True):
    if fast:
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"
        transfer = 'fast'
    else:
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "0"
        transfer = "slow"

    logger.info(f'Doing {transfer} download of {repo_id} to {local_dir}. Symlinks = {local_dir_use_symlinks}')
    if cache_dir is not None:
        os.environ['HF_HOME'] = cache_dir
        logger.info(f'Cache dir set to {cache_dir}')

    start_time = time.time()

    try:
        if not os.path.isdir(local_dir):
            logger.info(f"Creating {local_dir}")
            os.makedirs(local_dir, exist_ok=True)

        queue = Queue()
        p = Process(target=run_snapshot_download, args=(repo_id, local_dir, queue),
                    kwargs={'branch': branch, 'ignore_patterns': ignore_patterns, 'local_dir_use_symlinks': local_dir_use_symlinks, 'token': token})
        p.start()

        t = RepeatTimer(log_period, log_size, [start_time, local_dir, repo_id],
                        {'symlinks': local_dir_use_symlinks})
        t.start()

        p.join() # Wait for download to complete
        end_time = time.time()
        t.cancel()  # This cancels the Timer, ending the log_size calls

        result = queue.get()
        if result:
            logger.info("Download complete")
        else:
            logger.info("Download FAILED")
        seconds = end_time - start_time
        size, size_MB, size_GB = get_size(local_dir, repo_id, symlinks=local_dir_use_symlinks)
        if not fast:
            logger.info('\n\n\n')
        logger.info(f'Downloaded {size} bytes ({size_GB:.2f} GB) in {seconds:.2f}s, a rate of: {size_MB / seconds:.2f} MB/s')

        t.join() # make sure the timer is done
        
        return result
    except Exception as e:
        logger.info(f'Got exception: {e}')
        logger.info('Failed to download')
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hub download')
    parser.add_argument('repo', type=str, help='Repo name')
    parser.add_argument('model_folder', type=str, help='Model folder')
    parser.add_argument('--log_every', type=int, default=15, help='Log download progress every N seconds')
    parser.add_argument('--cache_dir', type=str, help='Set the HF cache folder')
    parser.add_argument('--branch', type=str, default="main", help='Branch to download from')
    parser.add_argument('--token', type=str, help='Use custom token')
    parser.add_argument('--symlinks', type=str, choices=['true', 'false', 'auto'], default="auto", help='Set to download to cache dir and symlink to target folder')
    parser.add_argument('--fast', '-f', type=str, default="1", help='Set to 1 to download fast (HF_HUB_ENABLE_HF_TRANSFER)')
    parser.add_argument('--ignore', '-i', nargs='+', type=str, help='patterns to ignore')
    args = parser.parse_args()

    token = args.token or True
    if hf_snapshot_download(args.repo, args.model_folder,
                            cache_dir=args.cache_dir,
                            branch=args.branch,
                            fast=args.fast,
                            log_period=args.log_every,
                            token=args.token,
                            ignore_patterns=args.ignore,
                            local_dir_use_symlinks=args.symlinks):
        logger.info("Downloaded successfully")
        sys.exit(0)
    else:
        logger.info("Downloaded failed")
        sys.exit(1)
