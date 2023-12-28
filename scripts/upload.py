from huggingface_hub import HfApi
import sys


def upload_model(model_dir, repo_id, repo_type):
    print(f'uploading {model_dir} to {repo_id}')
    api = HfApi()
    api.upload_folder(folder_path=model_dir, repo_id=repo_id, repo_type=repo_type)


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ['dataset', 'model']:
        print(f'usage: python upload.py local_path remote_repo_id repo_type, repo_type must be "dataset" or "model"')
        exit(-1)
    upload_model(sys.argv[1], sys.argv[2], sys.argv[3])
