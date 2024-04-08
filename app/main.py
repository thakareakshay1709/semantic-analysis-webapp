"""
This python module contains mainly the routing functions of FastApi endpoints.
"""

from typing import List

from fastapi import FastAPI, File, UploadFile

# from models
from app.ml.models import extract_semantic_features, get_semantic_similarity
from app.ml.utils import read_image

webapp = FastAPI()


@webapp.get("/")
async def health_endpoint():
    """
    This get function is designed to validate whether the endpoint is live.
    :return: dict message
    """
    print("Health endpoint")
    return {"message": "Hello! API is working fine!"}


@webapp.post("/densefeatureextraction/")
async def get_semantic_similarity_resnet(input_files: List[UploadFile] = File(...), ref_file: UploadFile = File(...)):
    """
    This post method determines semantic similarity using ResNet (tensors).
    :param input_files: list of input files
    :param ref_file: UploadFile type single reference image input
    :return: response: dict
    """
    # preprocess image
    input_content = {}
    for input_img in input_files:
        ip_img_content = await input_img.read()
        input_content[input_img.filename] = read_image(ip_img_content)
        print(f"Input file {input_img.filename} saved.")

    ref_content = {}
    ref_img_content = await ref_file.read()
    ref_content[ref_file.filename] = read_image(ref_img_content)
    print(f"Reference file {ref_file.filename} saved.")

    response = extract_semantic_features(input_content, ref_content)
    print(f"Response returned : {response}")
    return {"response": response}


@webapp.post("/simplefeatureextraction/")
async def get_semantic_similarity_with_vgg(input_files: List[UploadFile] = File(...), ref_file: UploadFile = File(...)):
    """
    This post method determines semantic similarity using VGG (torch).
    :param input_files: list of UploadFile type image input
    :param ref_file: UploadFile type single reference image input
    :return: response: dict
    """
    # print the summary of the model's architecture.
    print("Starting to process images and similarity with reference image.")

    # preprocess image
    input_content = {}
    for input_img in input_files:
        ip_img_content = await input_img.read()
        input_content[input_img.filename] = read_image(ip_img_content)
        print(f"Input file {input_img.filename} saved.")

    ref_content = {}
    ref_img_content = await ref_file.read()
    ref_content[ref_file.filename] = read_image(ref_img_content)
    print(f"Reference file {ref_file.filename} saved.")

    response = get_semantic_similarity(input_content, ref_content)
    print(f"Response returned : {response}")

    return {"response": response}
