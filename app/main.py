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
    This post method determines semantic similarity using Pytorch
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
async def get_semantic_similarity_with_vgg(
    input_images: List[UploadFile] = File(...), ref_image: UploadFile = File(...)
):
    """

    :param input_images: list of UploadFile type image input
    :param ref_image: UploadFile type single reference image input
    :return: response: dict
    """
    # print the summary of the model's architecture.
    print("Starting to process images and similarity with reference image.")
    _res = {}

    # preprocess image
    input_content = {}
    for input_img in input_images:
        ip_img_content = await input_img.read()
        input_content[input_img.filename] = read_image(ip_img_content)
        print(f"Input file {input_img.filename} saved.")

    ref_content = {}
    ref_img_content = await ref_image.read()
    ref_content[ref_image.filename] = read_image(ref_img_content)
    print(f"Reference file {ref_image.filename} saved.")

    response = get_semantic_similarity(input_content, ref_content)
    print(f"Response returned : {response}")

    # content1 = await input_images[0].read()
    # img1 = Image.open(BytesIO(content1))
    # resized_image1 = resize_image(img1)
    #
    # image_array1 = get_image_array(resized_image1)
    # print("image array", image_array1)
    # image_embedding1 = get_embeddings(image_array1)
    # print("embedding1", len(image_embedding1[0]))
    # embed1 = image_embedding1.flatten().tolist()
    #
    # _res["embed1"] = embed1
    # _res["file1"] = input_images[0].filename
    #
    # # image2
    # content2 = await input_images[1].read()
    # img2 = Image.open(BytesIO(content2))
    # resized_image2 = resize_image(img2)
    #
    # image_array2 = get_image_array(resized_image2)
    # print("image array", image_array2)
    # image_embedding2 = get_embeddings(image_array2)
    # print(type(image_embedding2))
    # print("embedding2", len(image_embedding2[0]))
    # _res["file2"] = input_images[1].filename
    # embed2 = image_embedding2.flatten().tolist()
    #
    # _res["embed2"] = embed2
    #
    # # similarity_score = cosine_similarity(image_embedding1, image_embedding2)  # .reshape(1, )
    # similarity_score = check_similarity(image_embedding1, image_embedding2)
    # # res["sim"] = similarity_score
    # print("similarity score")
    # # _res["score"] = similarity_score[0]
    # score = float(similarity_score[0])
    # _res["score"] = score
    # print(similarity_score[0], type(score))

    return {"response": response}
