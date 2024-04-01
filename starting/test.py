from PIL import Image

image_path = '/home/rinzler/Github/Image-Text-Matching/data/images/3637013_c675de7705.jpg'

try:
    with Image.open(image_path) as img:
        print(f"Successfully opened {image_path}")
except IOError:
    print(f"Cannot open {image_path}")
