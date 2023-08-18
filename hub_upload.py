#
# Upload a model to Hugging Face Hub
# Run with HF_HUB_ENABLE_HF_TRANSFER=1 for high speed upload
#

from huggingface_hub import HfApi, upload_file
import argparse
import time
import os
import pathlib

parser = argparse.ArgumentParser(description='hub upload')
parser.add_argument('repo', type=str, help='Repo name')
parser.add_argument('file_or_folder', type=str, help='Model folder')
parser.add_argument('--branch', type=str, help='Use and create branch')
parser.add_argument('--commit', type=str, default='Upload folder using huggingface_hub', help='Commit message')
parser.add_argument('--no_upload', action="store_true", help='Just make branch')
parser.add_argument('--upload_files', action="store_true", help='Upload file by file')

args = parser.parse_args()

file_or_folder = args.file_or_folder
repo = args.repo
branch = args.branch

api = HfApi(token = True)

if branch:
    print(f"Creating and using branch {branch}")
    try:
        api.create_branch(repo_id=repo,
            repo_type="model",
            branch=branch)
    except:
        print("Couldn't create branch - probably already exists?")
else:
    branch = "main"

if not args.no_upload:
    retry=True
    count=0
    while retry and count < 10:
        try:
            count += 1
            start_time = time.time()
            if os.path.isdir(file_or_folder):
                if not args.upload_files:
                    print(f"Uploading folder {file_or_folder} to {repo}")
                    api.upload_folder(
                        folder_path=file_or_folder,
                        path_in_repo="/",
                        repo_id=repo,
                        repo_type="model",
                        commit_message=args.commit,
                        revision=branch
                    )
                else:
                    for filename in os.listdir(file_or_folder):
                        file = os.path.join(file_or_folder, filename)
                        print(f"Uploading file {filename} to {repo}")
                        api.upload_file(
                            path_or_fileobj=file,
                            path_in_repo=f"/{filename}",
                            repo_id=repo,
                            repo_type="model",
                            commit_message=args.commit,
                            revision=branch
                        )
            elif os.path.isfile(file_or_folder):
                file = os.path.basename(file_or_folder)
                print(f"Uploading file {file_or_folder} to {repo}")
                api.upload_file(
                    path_or_fileobj=file_or_folder,
                    path_in_repo=f"/{file}",
                    repo_id=repo,
                    repo_type="model",
                    commit_message=args.commit,
                    revision=branch
                )
            else:
                raise FileNotFoundError(f"No folder or file found at {file_or_folder}")
                retry = False
            print(f"Total time to upload: {time.time() - start_time:.2f}")
            retry = False
        except Exception as e:
            print(f"Exception {e} occurred")
            retry=True
