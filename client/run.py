"""This module runs the client script with local test data."""
import sys

import requests
from pathlib import Path
import os
from os import path
import argparse


# preparing payload
def prepare_local_payload():
    """
    This function prepares payload to invoke endpoint with some local image hardcoded files.
    :return: headers, files object
    """
    test_data = Path(path.join(base_path, 'local_data', 'test_data'))

    headers = {
        'accept': 'application/json',
        # requests won't add a boundary if this header is set when you pass files=
        # 'Content-Type': 'multipart/form-data',
    }

    # you can add more test files into /local_data/test_data directory and add path below to run the analysis
    files = [
        ('input_files', ('c1.png', open(path.join(test_data, 'c1.png'), 'rb'), 'image/png')),
        ('input_files', ('d1.png', open(path.join(test_data, 'd1.png'), 'rb'), 'image/png')),
        ('ref_file', ('c1.png', open(path.join(test_data, 'c1.png'), 'rb'), 'image/png')),
    ]
    return headers, files


def invoke_dense_endpoint():
    """
    Invokes /densefeatureextraction/ endpoint to fetch response from ResNet model.
    """
    headers, files = prepare_local_payload()
    try:
        url = "http://127.0.0.1:8000/densefeatureextraction/"
        # endpoint getting response from ResNet CNN model
        response_dense = requests.post(url, headers=headers, files=files)
        print(f"Received response from {url}")
        print(response_dense.json())
    except Exception as e:
        print(f"Something went wrong with dense endpoint. {str(e)}")


def invoke_simple_endpoint():
    """
    Invokes /simplefeatureextraction/ endpoint to fetch response from VGG16 model.
    """
    headers, files = prepare_local_payload()
    try:
        url = "http://127.0.0.1:8000/simplefeatureextraction/"
        # endpoint getting response from VGG16 CNN model
        response_simple = requests.post(url, headers=headers, files=files)
        print(f"Received response from {url}")
        print(response_simple.json())
    except Exception as e:
        print(f"Something went wrong with simple endpoint. {str(e)}")


if __name__ == "__main__":
    # options = {1: "dense", 2: "simple"}
    base_path = Path.cwd().parent.absolute()
    # should be 'your-path/semantic-analysis-webapp' i.e root directory of this application
    print(f"Setting the working directory to {base_path}")
    os.chdir(base_path)

    # adding parser to serve user with option for CLI command
    parser = argparse.ArgumentParser()
    parser.add_argument("options")
    args = parser.parse_args()

    option = args.options

    if option == "dense":
        invoke_dense_endpoint()
    elif option == "simple":
        invoke_simple_endpoint()
    else:
        print("Invalid option.\nChoose 1) simple or 2) dense")
        sys.exit()
