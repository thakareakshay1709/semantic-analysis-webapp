"""This module provides common helper functions for machine learning tasks such as preprocessing."""

from io import BytesIO

import numpy as np
from keras.preprocessing import image
from PIL import Image


def get_image_array(_img) -> np.ndarray:
    """
    This function converts image array into 3D numpy array.
    :param _img: raw image numpy array
    :return: 3d converted numpy array
    """
    img_array = np.expand_dims(image.img_to_array(_img), axis=0)
    return img_array


def resize_image(_img, size=(224, 224)) -> np.ndarray:
    """
    This function resize image numpy array to desired size.
    :param _img: numpy image array
    :param size: tuple with size
    :return: numpy resized array
    """
    return _img.resize(size)


def read_image(byte_content: bytes) -> Image:
    """
    This function reads byte-type object into image.
    :param byte_content: bytes content read from raw image
    :return: Image type object
    """
    _img = Image.open(BytesIO(byte_content))
    return _img
