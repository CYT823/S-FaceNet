from PIL import Image, ImageEnhance
from tqdm import tqdm
import os

if __name__ == "__main__":
    source_root = "./CASIA_WebFace_Aligned"
    dest_root = "./CASIA_WebFace_Aligned_Augmentation"

    # Create new root folder
    if not os.path.isdir(dest_root): 
        os.mkdir(dest_root)

    for subfolder in tqdm(os.listdir(source_root)): 
        sour_subfolder_path = os.path.join(source_root, subfolder)
        dest_subfolder_path = os.path.join(dest_root, subfolder)

        # Create folder for new images
        if not os.path.isdir(dest_subfolder_path):
            os.mkdir(dest_subfolder_path)

        # Start processing...
        for img_name in os.listdir(sour_subfolder_path):
            img_path = os.path.join(sour_subfolder_path, img_name)

            print("Processing\t{}".format(img_path))
            img = Image.open(img_path)
            
            # Save original
            img.save(os.path.join(dest_subfolder_path, img_name))

            # Adjust image brightness
            enhancer = ImageEnhance.Brightness(img)
            # Darken image
            factor = 0.8
            output_img = enhancer.enhance(factor)
            output_img_name = '.'.join(img_name.split('.')[:-1]) + '_dar1.' +  img_name.split('.')[-1]
            output_img.save(os.path.join(dest_subfolder_path, output_img_name))
            factor = 0.5
            output_img = enhancer.enhance(factor)
            output_img_name = '.'.join(img_name.split('.')[:-1]) + '_dar2.' +  img_name.split('.')[-1]
            output_img.save(os.path.join(dest_subfolder_path, output_img_name))
            # Brightens image
            factor = 1.2 
            output_img = enhancer.enhance(factor)
            output_img_name = '.'.join(img_name.split('.')[:-1]) + '_bri1.' +  img_name.split('.')[-1]
            output_img.save(os.path.join(dest_subfolder_path, output_img_name))
            factor = 1.5
            output_img = enhancer.enhance(factor)
            output_img_name = '.'.join(img_name.split('.')[:-1]) + '_bri2.' +  img_name.split('.')[-1]
            output_img.save(os.path.join(dest_subfolder_path, output_img_name))

            # Flip imale gorizontal
            output_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            output_img_name = '.'.join(img_name.split('.')[:-1]) + '_flip.' +  img_name.split('.')[-1]
            output_img.save(os.path.join(dest_subfolder_path, output_img_name))

            # Adjust image sharpness
            enhancer = ImageEnhance.Sharpness(img)
            # Sharp image
            factor = 5
            output_img = enhancer.enhance(factor)
            output_img_name = '.'.join(img_name.split('.')[:-1]) + '_sharp.' +  img_name.split('.')[-1]
            output_img.save(os.path.join(dest_subfolder_path, output_img_name))
            # Blur image
            factor = 0.05
            output_img = enhancer.enhance(factor)
            output_img_name = '.'.join(img_name.split('.')[:-1]) + '_blur.' +  img_name.split('.')[-1]
            output_img.save(os.path.join(dest_subfolder_path, output_img_name))

            # Adjust image contrast
            enhancer = ImageEnhance.Contrast(img)
            # Decrease constrast
            factor = 0.5 
            output_img = enhancer.enhance(factor)
            output_img_name = '.'.join(img_name.split('.')[:-1]) + '_lowcon.' +  img_name.split('.')[-1]
            output_img.save(os.path.join(dest_subfolder_path, output_img_name))
            # Increase contrast
            factor = 1.5 
            output_img = enhancer.enhance(factor)
            output_img_name = '.'.join(img_name.split('.')[:-1]) + '_highcon.' +  img_name.split('.')[-1]
            output_img.save(os.path.join(dest_subfolder_path, output_img_name))

            # Rotate image
            angle = 5
            output_img = img.rotate(angle)
            output_img_name = '.'.join(img_name.split('.')[:-1]) + '_rotate1.' +  img_name.split('.')[-1]
            output_img.save(os.path.join(dest_subfolder_path, output_img_name))
            angle = 10
            output_img = img.rotate(angle)
            output_img_name = '.'.join(img_name.split('.')[:-1]) + '_rotate2.' +  img_name.split('.')[-1]
            output_img.save(os.path.join(dest_subfolder_path, output_img_name))
            angle = -5
            output_img = img.rotate(angle)
            output_img_name = '.'.join(img_name.split('.')[:-1]) + '_rotate3.' +  img_name.split('.')[-1]
            output_img.save(os.path.join(dest_subfolder_path, output_img_name))
            angle = -10
            output_img = img.rotate(angle)
            output_img_name = '.'.join(img_name.split('.')[:-1]) + '_rotate4.' +  img_name.split('.')[-1]
            output_img.save(os.path.join(dest_subfolder_path, output_img_name))