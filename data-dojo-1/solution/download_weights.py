from huggingface_hub import snapshot_download


REPO_ID = "google/gemma-4-E2B-it"
LOCAL_DIR = "weights"


def main():
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
    )
    print(f"weights downloaded to: {path}")


if __name__ == "__main__":
    main()
