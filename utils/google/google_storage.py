import glob
import os
import sys

import google
from google.cloud import storage


def upload_file_remote(path: str, bucket_name: str = 'forest-biomass') -> str:
    """
    Upload file and return path to file on Google Cloud Storage
    :param path: Path to local file (to be uploaded)
    :param bucket_name: Name of bucket on GCS
    :return: Path of remote file on GCS
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} could not be found")

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(path.split('/')[-1])

    blob.upload_from_filename(path)

    return f"gs://{bucket_name}/{path.split('/')[-1]}"


def delete_file_remote(path: str, bucket_name: str = 'forest-biomass') -> None:
    """
    Method to delete
    :param path: Path to file on GCS to be deleted
    :param bucket_name: Name of GCS bucket
    :return: None
    """
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(path.split('/')[-1])
    blob.delete()


def upload_local_directory_to_gcs(path: str, bucket, gcs_path: str):
    """
    Upload folder to Google Cloud Storage
    :param path: Path to local folder (to be uploaded)
    :param bucket: Bucket on GCS
    :param gcs_path: Path to upload a certain folder to on GCS
    """
    assert os.path.isdir(path)
    for local_file in glob.glob(path + '/**'):
        if not os.path.isfile(local_file):
            upload_local_directory_to_gcs(local_file, bucket, gcs_path + "/" + os.path.basename(local_file))
        else:
            remote_path = gcs_path + "/" + local_file[1 + len(path):]
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)


def authenticate_remote():
    """
    Method to authenticate to GSC
    :return: None or error
    """
    if sys.platform == "linux":
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "../../keys/forestbiomass-key-epoch.json"
    elif sys.platform.startswith("win"):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "../../keys/forestbiomass-key-epoch.json"

    try:
        storage.Client()
        print("Authenticated successfully")
        return

    except google.auth.exceptions.DefaultCredentialsError as e:
        return e


if __name__ == "__main__":
    authenticate_remote()

    storage_client = storage.Client()

    bucket = storage_client.bucket('forest-biomass')

    upload_local_directory_to_gcs(r"C:\Users\Team Epoch A\Desktop\forest-biomass", bucket, 'forest')
