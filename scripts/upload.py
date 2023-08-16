from huggingface_hub import HfApi
import sys


def upload_model(model_dir, repo_id):
    print(f'uploading {model_dir} to {repo_id}')
    api = HfApi()
    api.upload_folder(folder_path=model_dir, repo_id=repo_id)


if __name__ == '__main__':
    upload_model(sys.argv[1], sys.argv[2])
