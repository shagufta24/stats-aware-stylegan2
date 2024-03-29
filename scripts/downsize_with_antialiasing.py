from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument("--path", type=str, default='', help="Source image directory")
parser.add_argument("--out", type=str, default='', help="Destination image directory")
parser.add_argument("--out_size", type=int, default=128, help="New size for e.g. 128 for 128 x 128")
args = parser.parse_args()
if args.path == '':
    print("Please run script again while specifying the correct path to input image directory")
else:
    if args.out == '':
        args.out == args.path
        args.out += '/downsized'
    os.makedirs(args.out, exist_ok=True)
    images = [file for file in os.listdir(args.path) if file.endswith(('.jpg', '.jpeg', '.png'))]
    for image_name in images:
        image_path = os.path.join(args.path, image_name)
        image = Image.open(image_path)
        
        # Define the new size for downsampling
        new_size = (args.out_size, args.out_size)  # Adjust as needed
        
        # Resize the image using Lanczos interpolation
        resized_image = image.resize(new_size, Image.LANCZOS)
        
        # Save the resized image
        resized_image_path = os.path.join(args.out, image_name)
        resized_image.save(resized_image_path)
