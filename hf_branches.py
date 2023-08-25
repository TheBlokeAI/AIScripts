#
# Simple script for listing, creating and deleting branches on Huggingface repos
# Create and delete operations require an HF access token
# By default the scripts assumes you are logged in to Huggingface Hub with `huggingface-cli login`. Alternatively you can pass a token with --token
#

import argparse
from huggingface_hub import HfApi

def create_branch(api, repo_id, branch, quiet=False):
    try:
        if branch not in get_branches(api, repo_id):
            api.create_branch(repo_id=repo_id, branch=branch)
            if not quiet:
                print(f"Successfully created branch '{branch}' in repo '{repo_id}'.")
        else:
            print(f"Branch '{branch}' already exists in repo '{repo_id}'.")
    except Exception:
        print(f"Error creating branch '{branch}' on '{repo_id}'.")
        raise

def delete_branch(api, repo_id, branch, quiet=False):
    try:
        if branch in get_branches(api, repo_id):
            api.delete_branch(repo_id=repo_id, branch=branch)
            if not quiet:
                print(f"Successfully deleted branch '{branch}' in repo '{repo_id}'.")
        else:
            if not quiet:
                print(f"Branch '{branch}' does not exist in repo '{repo_id}'.")
    except Exception:
        print(f"Error deleting branch '{branch}' on '{repo_id}'.")
        raise

def list_branches(api, repo_id, quiet=False):
    branch_list = get_branches(api, repo_id)
    if not quiet:
        print(f"Branches in {repo_id}:")
    for branch in branch_list:
        if not quiet:
            print(f" * {branch}")
        else:
            print(branch)

def get_branches(api, repo_id):
    try:
        branches = api.list_repo_refs(repo_id).branches
        return [branch.name for branch in branches]
    except Exception:
        print(f"Error getting branches on '{repo_id}'.")
        raise

def main():
    parser = argparse.ArgumentParser(description="Manage branches in HuggingFace Hub. Create and delete operations require login with `huggingface-cli login`, or passing a token with `--token`.")
    parser.add_argument("command", choices=['create', 'delete', 'list'], help="Command to execute.")
    parser.add_argument("repo_id", help="Repository ID.")
    parser.add_argument("branch", nargs='?', default=None, help="Branch name (for create/delete commands).")
    parser.add_argument("--token", type=str, help="Use to specify an HF token. Otherwise it is assumed you are already logged into to HF using `huggingface-cli login`")
    parser.add_argument('-q', "--quiet", action="store_true", help="Don't output anything (except necessary text)")

    args = parser.parse_args()

    try:
        api = HfApi(token=args.token or True)
        try:
            if args.command == "create":
                if not args.branch:
                    print("Error: Branch name required for create command.")
                    return
                create_branch(api=api, repo_id=args.repo_id, branch=args.branch, quiet=args.quiet)
            elif args.command == "delete":
                if not args.branch:
                    print("Error: Branch name required for delete command.")
                    return
                delete_branch(api=api, repo_id=args.repo_id, branch=args.branch, quiet=args.quiet)
            elif args.command == "list":
                list_branches(api=api, repo_id=args.repo_id, quiet=args.quiet)
        except Exception as e:
            print(f"Exception while running command '{args.command}': {e}")
            print("Cannot continue.")
    except Exception as e:
        print(f"Exception logging in to Huggingface Hub: {e}")
        print("Cannot continue.")

if __name__ == "__main__":
    main()
