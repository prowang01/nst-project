from api.transfer_style import transfer_style
import matplotlib.pylab as plt

if __name__ == "__main__":
    model_path = "model/magenta_arbitrary-image-stylization-v1-256_2"
    content_image_path = "assets/content.jpg"
    style_image_path = "assets/style_hockney.jpg"

    img = transfer_style(content_image_path, style_image_path, model_path)

    plt.imsave("stylized_image.jpeg", img)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
