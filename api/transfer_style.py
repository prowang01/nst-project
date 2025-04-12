import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def transfer_style(content_image_path, style_image_path, model_path, max_dim=512):
    """
    Apply neural style transfer on two images with optional resizing to save memory.

    :param content_image_path: path to the content image
    :param style_image_path: path to the style image
    :param model_path: path to the downloaded pre-trained model (TF Hub)
    :param max_dim: max dimension to resize content image (default 512)
    :return: stylized image as a 3D numpy array
    """
    from PIL import Image

    def load_and_process_image(path, resize=True):
        img = Image.open(path).convert('RGB')
        if resize:
            img.thumbnail((max_dim, max_dim))
        img = np.array(img).astype(np.float32)[np.newaxis, ...] / 255.
        return tf.constant(img)

    print("Loading and resizing images...")
    content_image = load_and_process_image(content_image_path, resize=True)
    style_image = load_and_process_image(style_image_path, resize=True)
    style_image = tf.image.resize(style_image, (256, 256))

    print("Loading pre-trained model...")
    hub_module = hub.load(model_path)

    print("Generating stylized image now...")
    outputs = hub_module(content_image, style_image)
    stylized_image = outputs[0]

    print("Stylizing completed.")
    return stylized_image[0].numpy()  # Remove batch dimension
