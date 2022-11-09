import os
import sys

import google
from google.cloud import storage


def upload_file_remote(path: str, bucket_name: str) -> str:
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


def delete_file_remote(path: str, bucket_name: str) -> None:
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


def authenticate_remote():
    """
    Method to authenticate to GSC
    :return: None or error
    """
    if sys.platform == "linux":
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = None  # TODO: ADD PATH TO GOOGLE STORAGE BUCKET API KEY
    elif sys.platform.startswith("win"):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = None  # TODO: ADD PATH TO GOOGLE STORAGE BUCKET API KEY

    try:
        storage.Client()
        return

    except google.auth.exceptions.DefaultCredentialsError as e:
        return e
