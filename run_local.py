from api.transfer_style import transfer_style
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

if __name__ == "__main__":
    model_path = "model/magenta_arbitrary-image-stylization-v1-256_2"
    content_image_path = "assets/content.jpg"
    style_image_path = "assets/style_hockney.jpg"

    # Générer image stylisée
    img = transfer_style(content_image_path, style_image_path, model_path)

    # Charger les images d'origine pour l'affichage
    content_img = Image.open(content_image_path).resize((256, 256))
    style_img = Image.open(style_image_path).resize((256, 256))
    stylized_img = Image.fromarray((img * 255).astype(np.uint8)).resize((256, 256))

    # Affichage côte à côte
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Content", "Style", "Stylized"]
    images = [content_img, style_img, stylized_img]

    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("comparison.jpg")
    plt.show()
