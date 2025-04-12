import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def transfer_style(content_image, style_image, model_path):
    """
    :param content_image: path of the content image
    :param style_image: path of the style image
    :param model_path: path to the downloaded pre-trained model.
    :return: stylized image as a 3D numpy array
    """

    print("Loading images...")
    content_image = plt.imread(content_image)
    style_image = plt.imread(style_image)

    print("Resizing and Normalizing images...")
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = tf.image.resize(style_image, (256, 256))

    print("Loading pre-trained model...")
    hub_module = hub.load(model_path)

    print("Generating stylized image now...")
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]

    stylized_image = np.array(stylized_image)
    stylized_image = stylized_image.reshape(
        stylized_image.shape[1], stylized_image.shape[2], stylized_image.shape[3])

    print("Stylizing completed.")
    return stylized_image
