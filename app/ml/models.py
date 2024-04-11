"""This python module contains methods to load and tune pytorch & tensorflow pretrained models."""

from typing import Dict, List

import numpy as np
from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import ResNet50_Weights, resnet50

from app.ml.utils import get_image_array, resize_image


class ResNet:
    """This class instantiate ResNet model."""

    def __init__(self):
        print("Initiating ResNet model")
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights)
        self.model.eval()


resnet = ResNet()
res_model = resnet.model


class VGGModel:
    """This class instantiate VGG model."""

    def __init__(self):
        print("Initiating VGG model")
        self.model = VGG16(weights="imagenet", include_top=False, pooling="max", input_shape=(224, 224, 3))


# instantiating class and model instance
vgg_class = VGGModel()
vgg16 = vgg_class.model
for model_layer in vgg16.layers:
    model_layer.trainable = False


# def initiate_model():
#     """
#     This function initializes pretrained VGG16 model with its config.
#     :return: VGG model instance
#     """
#     print("Initiating model")
#     # TODO: Try to load this only once
#     vgg16 = VGG16(weights="imagenet", include_top=False, pooling="max", input_shape=(224, 224, 3))
#
#     # print("summary")
#     # vgg16.summary()
#     for model_layer in vgg16.layers:
#         model_layer.trainable = False
#     return vgg16
#
#
# vgg16 = initiate_model()


def get_embeddings(_img_array: np.ndarray) -> np.ndarray:
    """
    This function calls the predict method of VGG16 pretrained model and get its embeddings.
    :param _img_array: ndarray of image
    :return: ndarray image embeddgings
    """
    print("Generating image embeddings")
    embeddings = vgg16.predict(_img_array)
    return embeddings


def check_similarity(input_image_embed: np.ndarray, ref_image_embed: np.ndarray) -> float:
    """
    This function determines the cosine similarity of two numpy arrays.
    :param input_image_embed: ndarray embeddings of input image
    :param ref_image_embed: ndarray embeddings of reference image
    :return: cosine similarity score
    """
    _cosine_score = cosine_similarity(input_image_embed, ref_image_embed)
    return round(float(_cosine_score[0]), 4)


def get_semantic_similarity(input_content: Dict, ref_content: Dict) -> List:
    """
    This function extracts features of image and get their cosine similarity.
    :param input_content: dict of one or more input filenames as key and it's content as value
    :param ref_content: dict of one reference filename as key and it's content as value
    :return: list of record as dict objects with one input file, reference file and their cosine score
    """

    print(f"Processing {len(input_content)} input images with {len(ref_content)} reference image.")
    # lowest_similarity = 0.0
    response = []
    # processing reference image
    ref_name, ref_img_content = list(ref_content.keys())[0], list(ref_content.values())[0]
    ref_resized = resize_image(ref_img_content)
    ref_array = get_image_array(ref_resized)
    ref_embeds = get_embeddings(ref_array)
    print("Processed reference image.")

    # processing input images
    for filename, img_content in input_content.items():
        print(f"Processing file {filename}")
        record = {}
        resized = resize_image(img_content)
        array = get_image_array(resized)
        embeds = get_embeddings(array)
        semantic_score = check_similarity(embeds, ref_embeds)
        print(f"Processed {filename} with score {semantic_score}")
        record["input_filename"] = filename
        record["ref_filename"] = ref_name
        record["cosine_similarity"] = semantic_score
        record["features length"] = len(embeds[0])
        record_input_embeds = embeds[0].tolist()
        record_ref_embeds = ref_embeds[0].tolist()
        record["embeddings"] = {"input_embeds": record_input_embeds, "ref_embeds": record_ref_embeds}
        response.append(record.copy())
    print("Preparing response")
    return response


def extract_semantic_features(input_content: Dict, ref_content: Dict):
    """
    This function extracts features of image and get their cosine similarity.
    :param input_content: dict of one or more input filenames as key and it's content as value
    :param ref_content: dict of one reference filename as key and it's content as value
    :return: list of record as dict objects with one input file, reference file and their cosine score
    """
    print(f"Processing {len(input_content)} input images with {len(ref_content)} reference image.")

    ref_name, ref_img_content = list(ref_content.keys())[0], list(ref_content.values())[0]

    preprocess = resnet.weights.transforms(antialias=True)

    batch_ref = preprocess(ref_img_content).unsqueeze(0)
    ref_pred = res_model(batch_ref).squeeze(0).softmax(0)
    ref_class = ref_pred.argmax().item()
    ref_independant_score = round(ref_pred[ref_class].item(), 4)
    ref_cat = resnet.weights.meta["categories"][ref_class]
    print("Processed reference image.")

    # processing input images
    response = []
    for filename, img_content in input_content.items():
        print(f"Processing file {filename}")
        _record = {}
        batch_input = preprocess(img_content).unsqueeze(0)
        input_pred = res_model(batch_input).squeeze(0).softmax(0)
        score = check_similarity([input_pred.detach().numpy()], [ref_pred.detach().numpy()])
        print(f"Processed {filename} with score {score}")
        input_class = input_pred.argmax().item()
        independant_input_score = round(input_pred[input_class].item(), 4)
        input_cat = resnet.weights.meta["categories"][input_class]
        _record["input_filename"] = filename
        _record["ref_filename"] = ref_name
        _record["cosine_similarity"] = score
        _record["independant_input_score"] = independant_input_score
        _record["ref_independant_score"] = ref_independant_score
        _record["input_category"] = input_cat
        _record["ref_category"] = ref_cat
        _record["features length"] = len(input_pred)
        # _record["embeddings"] = {"input_embeds": input_pred.tolist(), "ref_embeds": ref_pred.tolist()}
        response.append(_record.copy())
        print(f"Record for {filename} and {ref_name} added to response.")

    print("Preparing response")
    return response
