#
# Download a single file from Hugging Face hub
# This should probably be a method in hub_download.py instead
# NOTE: requires `pip3 install hf_transfer` for fast transfers (the default)
#

#TODO: disable fast downloads if hf_transfer is not installed

import logging
import time
import argparse
import os
import sys

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

def run_hub_download(repo_id, filename, local_dir, branch="main", token=True, local_dir_use_symlinks="auto"):
    from huggingface_hub import hf_hub_download
    retry = True
    try_count = 0
    max_tries = 5
    while retry and try_count < max_tries:
        try:
            hf_hub_download(token=token,
                            filename=filename,
                            repo_id=repo_id,
                            local_dir=local_dir,
                            revision=branch,
                            local_dir_use_symlinks=local_dir_use_symlinks)
            retry = False
        except Exception as e:
            logger.error(f"Exception: {e}")
            try_count += 1
            logger.info(f"Retrying {try_count} of {max_tries}")
            time.sleep(1)
    if retry:
        logger.error(f"Failed to download {repo_id}/{filename} after {max_tries} tries. Exiting.")
        return False
    else:
        return True

def hf_download(repo_id, filename, local_dir, branch="main", fast=True, cache_dir=None, local_dir_use_symlinks=False):
    if fast:
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"
        transfer = 'fast'
    else:
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "0"
        transfer = "slow"

    logger.info(f'Doing {transfer} download of {repo_id}/{filename} to {local_dir}. Symlinks = {local_dir_use_symlinks}')
    if cache_dir is not None:
        os.environ['HF_HOME'] = cache_dir
        logger.info(f'Cache dir set to {cache_dir}')

    start_time = time.time()

    try:
        return run_hub_download(repo_id, filename, local_dir, branch=branch, local_dir_use_symlinks=local_dir_use_symlinks)
    except Exception as e:
        logger.info(f'Got exception: {e}')
        logger.info('Failed to download')
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hub download')
    parser.add_argument('repo', type=str, help='Repo name')
    parser.add_argument('filename', type=str, help='Model folder')
    parser.add_argument('model_folder', type=str, help='Model folder')
    parser.add_argument('--cache_dir', type=str, help='Set the HF cache folder')
    parser.add_argument('--branch', type=str, default="main", help='Branch to download from')
    parser.add_argument('--symlinks', type=str, choices=['true', 'false', 'auto'], default="auto", help='Set to download to cache dir and symlink to target folder')
    parser.add_argument('--fast', '-f', type=str, default="1", help='Set to 1 to download fast (HF_HUB_ENABLE_HF_TRANSFER)')
    args = parser.parse_args()

    if hf_download(args.repo, args.filename, args.model_folder,
                            cache_dir=args.cache_dir,
                            branch=args.branch,
                            fast=args.fast,
                            local_dir_use_symlinks=args.symlinks):
        logger.info("Downloaded successfully")
        sys.exit(0)
    else:
        logger.info("Downloaded failed")
        sys.exit(1)
