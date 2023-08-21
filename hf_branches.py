#
# Simple script for listing, creating and deleting branches on Huggingface repos
# By default it requires that you are logged in to Huggingface Hub, or you can pass a token with --token
#

import argparse
from huggingface_hub import HfApi

def create_branch(api, repo_id, branch, exist_ok=False):
    try:
        if branch not in get_branches(api, repo_id):
            api.create_branch(repo_id=repo_id, branch=branch, exist_ok=exist_ok)
            print(f"Successfully created branch '{branch}' in repo '{repo_id}'.")
        else:
            print(f"Branch '{branch}' already exists in repo '{repo_id}'.")
    except Exception:
        print(f"Error creating branch '{branch}' on '{repo_id}'.")
        raise

def delete_branch(api, repo_id, branch):
    try:
        if branch in get_branches(api, repo_id):
            api.delete_branch(repo_id=repo_id, branch=branch)
            print(f"Successfully deleted branch '{branch}' in repo '{repo_id}'.")
        else:
            print(f"Branch '{branch}' does not exist in repo '{repo_id}'.")
    except Exception:
        print(f"Error deleting branch '{branch}' on '{repo_id}'.")
        raise

def list_branches(api, repo_id):
    branch_list = get_branches(api, repo_id)
    print(f"Branches in {repo_id}:")
    for branch in branch_list:
        print(f" * {branch}")

def get_branches(api, repo_id):
    try:
        branches = api.list_repo_refs(repo_id).branches
        return [branch.name for branch in branches]
    except Exception:
        print(f"Error getting branches on '{repo_id}'.")
        raise

def main():
    parser = argparse.ArgumentParser(description="Manage branches in HuggingFace Hub.")
    parser.add_argument("command", choices=['create', 'delete', 'list'], help="Command to execute.")
    parser.add_argument("repo_id", help="Repository ID.")
    parser.add_argument("branch", nargs='?', default=None, help="Branch name (for create/delete commands).")
    parser.add_argument("--exist_ok", action="store_true", help="Whether existing branch creation is okay (applies only to 'create' command).")
    parser.add_argument("--token", type=str, help="Use to specify an HF token. Otherwise it is assumed you are already logged into to HF using `huggingface-cli login`")

    args = parser.parse_args()

    try:
        api = HfApi(token=args.token or True)
        try:
            if args.command == "create":
                if not args.branch:
                    print("Error: Branch name required for create command.")
                    return
                create_branch(api=api, repo_id=args.repo_id, branch=args.branch, exist_ok=args.exist_ok)
            elif args.command == "delete":
                if not args.branch:
                    print("Error: Branch name required for delete command.")
                    return
                delete_branch(api=api, repo_id=args.repo_id, branch=args.branch)
            elif args.command == "list":
                list_branches(api=api, repo_id=args.repo_id)
        except Exception as e:
            print(f"Exception while running command '{args.command}': {e}")
            print("Cannot continue.")
    except Exception as e:
        print(f"Exception logging in to Huggingface Hub: {e}")
        print("Cannot continue.")

if __name__ == "__main__":
    main()
