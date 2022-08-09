import argparse
import os
from tqdm import tqdm
from PIL import Image

'''
This program is only resize the image into 112*112.
After doing this program, remember go "datasets" folder and execute "annotation_generation.py"
Start training...the new dataset is without doing any preprocess.
'''
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_root", help = "specify your source dir", default = "../datasets/VGG_Face2/data/train", type = str)
    parser.add_argument("-d", "--dest_root", help = "specify your destination dir", default = "../datasets/VGG_Face2_Resize", type = str)
    parser.add_argument("-c", "--crop_size", help = "specify crop size", default = 112, type = int)
    args = parser.parse_args()

    source_root = args.source_root  # specify your source dir
    dest_root = args.dest_root      # specify your destination dir
    crop_size = args.crop_size      # specify size of aligned faces, align and crop with padding
    if not os.path.isdir(dest_root): 
        os.mkdir(dest_root)
    
    for subfolder in tqdm(os.listdir(source_root)):
        if "CFP-FP" in source_root:
            sour_subfolder_path = os.path.join(source_root, subfolder, "frontal")
        else:
            sour_subfolder_path = os.path.join(source_root, subfolder)
        
        dest_subfolder_path = os.path.join(dest_root, subfolder)

        # Create folder for new images
        if not os.path.isdir(dest_subfolder_path):
            os.mkdir(dest_subfolder_path)

        # Start processing...
        for image_name in os.listdir(sour_subfolder_path):
            img_path = os.path.join(sour_subfolder_path, image_name)
            print("Processing\t{}".format(img_path))

            # Step 1. Open it
            img = Image.open(img_path)
            # Step 2. Resize it
            new_img = img.resize((crop_size, crop_size))
            # Step 3. Save it
            new_img.save(os.path.join(dest_subfolder_path, image_name))