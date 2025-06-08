import numpy as np
from PIL import Image, ImageDraw
import random
import os

os.makedirs("synthetic_dataset/images", exist_ok=True)

def generate_image(width=256, height=256, min_points=1, max_points=20, contrast_level=0.5):
    # Background
    base_color = int(128 * (1 + contrast_level))  # 0 to 255
    img = Image.new('L', (width, height), color=base_color)
    draw = ImageDraw.Draw(img)

    num_points = random.randint(min_points, max_points)
    for _ in range(num_points):
        x, y = random.randint(5, width-5), random.randint(5, height-5)
        r = 2  # radius
        color = 0 if random.random() > 0.5 else 255  # black or white point
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)

    return img

# Generate 100 images
for i in range(100):
    contrast = random.uniform(-0.4, 0.4)  # Vary contrast
    img = generate_image(contrast_level=contrast)
    img.save(f"synthetic_dataset/images/img_{i:03d}.png")
