from PIL import Image


def convert_grayscale_to_rgb(img: Image.Image) -> Image.Image:
    """
    Convert a grayscale image to RGB if necessary.

    Args:
        img (Image.Image): Input image.

    Returns:
        Image.Image: RGB image.
    """
    assert isinstance(
        img, Image.Image
    ), f"Expected img to be a PIL Image, but got {type(img)}"

    # Convert the image to RGB if it is not already in RGB mode
    if img.mode != "RGB":
        img = img.convert("RGB")

    return img
