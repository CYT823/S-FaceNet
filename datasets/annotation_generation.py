import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Annotation file generator')
parser.add_argument('--source', '-s', type=str, default='VGG_Face2_Resize', help='image data folder path')
parser.add_argument('--file', '-f',type=str, default='VGG_Face2_Resize/vgg_face2_resize_112.txt', help='anno file path')
args = parser.parse_args()

imgdir = args.source
list_txt_file = args.file
docs = [f for f in os.listdir(imgdir) if not f.startswith('.')]
docs.sort()

label = 0
for name in tqdm(docs):
    # # 去除 VGG-Face2 的資料
    # if name[0] == 'n': 
    #     continue
    
    print('writing folder:', name)
    image_folder = imgdir + '/' + name

    files = [f for f in os.listdir(image_folder) if not f.startswith('.')]
    files.sort()

    for file in files:
        txt_name = os.path.join(name, file)

        with open(list_txt_file, 'a') as f:
            f.write(txt_name+' '+str(label)+'\n')
        f.close()

    label+= 1

print('writing finished')


