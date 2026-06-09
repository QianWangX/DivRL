from PIL import Image

def extract_and_transform_pil(rgb_image, mask_image, scale, shift_x, shift_y):
    """
    rgb_image: PIL Image object (RGB)
    mask_image: PIL Image object (L or 1)
    scale: float (e.g., 1.5)
    shift_x, shift_y: int/float pixels
    """
    # 1. Ensure mask is in 'L' mode (grayscale) for transparency logic
    mask = mask_image.convert('L')
    
    # 2. Put object on white background
    # Create a solid white image of the same size
    white_bg = Image.new("RGB", rgb_image.size, (255, 255, 255))
    
    # Composite: Use mask to choose between the original image and white background
    extracted_img = Image.composite(rgb_image, white_bg, mask)

    # 3. Define the Affine Transformation
    # PIL uses the inverse matrix (mapping output to input).
    # To scale by 's' and shift by 't', the inverse parameters are:
    # (1/s, 0, -shift_x/s, 0, 1/s, -shift_y/s)
    inv_scale = 1.0 / scale
    matrix = (
        inv_scale, 0, -shift_x * inv_scale,
        0, inv_scale, -shift_y * inv_scale
    )
    
    # 4. Apply transformation to both
    # We use Image.BILINEAR for the image and Image.NEAREST for the mask to keep edges sharp
    mod_image = extracted_img.transform(
        rgb_image.size, Image.AFFINE, matrix, resample=Image.BILINEAR, fillcolor=(255, 255, 255)
    )
    
    mod_mask = mask.transform(
        mask.size, Image.AFFINE, matrix, resample=Image.NEAREST, fillcolor=0
    )

    return mod_image, mod_mask

# Example Usage:
# img = Image.open('image.jpg').convert('RGB')
# msk = Image.open('mask.png').convert('L')
# new_img, new_mask = extract_and_transform_pil(img, msk, 0.8, 100, 50)