from huggingface_hub import HfApi
import sys


def upload_file_func(file_path, repo_id, repo_type):
    print(f'uploading {file_path} to {repo_id}')
    api = HfApi()
    api.upload_file(path_or_fileobj=file_path, path_in_repo=file_path, repo_id=repo_id, repo_type=repo_type)


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ['dataset', 'model']:
        print(f'usage: python upload_file.py local_path remote_repo_id repo_type, repo_type must be "dataset" or "model"')
        exit(-1)
    upload_file_func(sys.argv[1], sys.argv[2], sys.argv[3])
