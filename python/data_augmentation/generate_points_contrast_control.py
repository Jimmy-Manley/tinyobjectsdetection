import numpy as np
from PIL import Image, ImageDraw
import random
import os

os.makedirs("synthetic_dataset/images", exist_ok=True)

def generate_image(width=256, height=256, min_points=1, max_points=20, contrast_level=0.5):
    """
    Generate a synthetic grayscale image with fixed background and controlled contrast small dots.
    contrast_level ∈ [-1, 1] controls how visible the objects are.
    """

    # Fixed background color: mid-gray
    base_color = 128
    img = Image.new('L', (width, height), color=base_color)
    draw = ImageDraw.Draw(img)

    num_points = random.randint(min_points, max_points)

    for _ in range(num_points):
        x, y = random.randint(5, width - 5), random.randint(5, height - 5)
        r = 2  # radius

        # Random direction: darker or lighter than background
        direction = random.choice([-1, 1])
        delta = int(contrast_level * 127)  # max possible difference from base
        object_color = max(0, min(255, base_color + direction * delta))

        draw.ellipse((x - r, y - r, x + r, y + r), fill=object_color)

    return img

# Generate 100 images with random contrast levels
for i in range(100):
    contrast = random.uniform(0.0, 1.0)  # 0 = invisible, 1 = high contrast
    img = generate_image(contrast_level=contrast)
    img.save(f"synthetic_dataset/images/img_{i:03d}.png")
