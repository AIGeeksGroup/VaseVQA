import os
import zipfile
from huggingface_hub import snapshot_download

def main():
    target_dir = "Data"
    os.makedirs(target_dir, exist_ok=True)

    print("Downloading VaseVQA dataset from HuggingFace...")
    snapshot_download(
        repo_id="AIGeeksGroup/VaseVQA",
        repo_type="dataset",
        local_dir=target_dir,
        local_dir_use_symlinks=False  # ensure real files, not symlinks
    )

    zip_path = os.path.join(target_dir, "images.zip")
    if os.path.exists(zip_path):
        print("Unzipping images.zip ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        os.remove(zip_path)
        print("Removed images.zip")

    print("Dataset downloaded and ready at ./Data")

if __name__ == "__main__":
    main()
