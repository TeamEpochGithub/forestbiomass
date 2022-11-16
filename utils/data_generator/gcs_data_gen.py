from google.api_core import page_iterator
import tensorflow as tf
from keras import datasets, layers, models
import google
from google.cloud import storage
import numpy as np
import sys
import os
from tensorflow.python.lib.io import file_io
from io import StringIO, BytesIO

# arr = np.load(BytesIO(file_io.read_file_to_string("gs://forest-biomass/forest/0003d2eb/0/S1/3.npy", binary_mode=True)))
# import os
# os.system('gsutil cp gs://forest-biomass/forest/0003d2eb/0/S1/2.npy ../../data')
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
authenticate_remote()
storage_client = storage.Client()
bucket = storage_client.bucket("forest-biomass")

blobs = storage_client.list_blobs("forest-biomass", prefix='forest')
# blobs = storage_client.list_blobs("forest-biomass")

for blob in blobs:
    print(blob.name)
    blob = BytesIO(blob.download_as_string())
    print(np.load(blob))
    break
# blob = bucket.blob(r"/5d8ffbe6/2/S1\0.npy")

def _item_to_value(iterator, item):
    return item
def list_directories(bucket_name, prefix):
    if prefix and not prefix.endswith('/'):
        prefix += '/'

    extra_params = {
        "projection": "noAcl",
        "prefix": prefix,
        "delimiter": '/'
    }

    gcs = storage.Client()

    path = "/b/" + bucket_name + "/o"

    iterator = page_iterator.HTTPIterator(
        client=gcs,
        api_request=gcs._connection.api_request,
        path=path,
        items_key='prefixes',
        item_to_value=_item_to_value,
        extra_params=extra_params,
    )

    return [x for x in iterator]

dir_list = list_directories('forest-biomass', 'forest')
print(dir_list)