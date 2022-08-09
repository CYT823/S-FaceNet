from PIL import Image
from detector import detect_faces
from align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
import os
from tqdm import tqdm
import argparse

''' Check landmarks
    return: True/False, bounding-boxes, landmarks
    bounding-box list: [left top x, left top y, right bottom x, right bottom y]
    landmark list: [left eye x, right eye x, nose x, lip left side x, lip right side x,
                left eye y, right eye y, nose y, lip left side y, lip right side y]
'''
def is_landmark_detected(img, img_path):
    try:
        bounding_boxes, landmarks = detect_faces(img)
    except Exception:
        print("{} is discarded due to exception!".format(img_path))
        return False, [], []

    # Discard those without landmark
    if len(landmarks) == 0: 
        print("{} is discarded due to non-detected landmarks!".format(img_path))
        return False, [], []
    
    return True, bounding_boxes, landmarks 


''' Find main face among multifaces in an image
    return: face_index, bounding-box, landmark (only one set)
'''
def get_main_face(bounding_boxes, landmarks, img_path):
    # Keep the largest face coordinate in the image
    maxWidth = 0
    seletedIndex = 0
    for i in range (0, len(bounding_boxes)):
        if bounding_boxes[i][2] - bounding_boxes[i][0] > maxWidth:
            maxWidth = bounding_boxes[i][2] - bounding_boxes[i][0]
            seletedIndex = i
    bounding_box = bounding_boxes[seletedIndex]
    landmark = landmarks[seletedIndex]

    return seletedIndex, bounding_box, landmark


''' Check overlap issue
    return: True/False
'''
def is_overlap(seletedIndex, bounding_boxes):
    # Remove images having serious overlap problem (Only consider width)
    for i in range (0, len(bounding_boxes)):
        if i == seletedIndex: 
            continue

        isOverlap = False
        if bounding_boxes[i][0] < bounding_boxes[seletedIndex][0] and bounding_boxes[i][2] > bounding_boxes[seletedIndex][0]:
            if bounding_boxes[i][2] - bounding_boxes[seletedIndex][0] > (bounding_boxes[seletedIndex][2] - bounding_boxes[seletedIndex][0])/4:
                isOverlap = True
                break
        if bounding_boxes[i][2] > bounding_boxes[seletedIndex][2] and bounding_boxes[seletedIndex][2] > bounding_boxes[i][0]:
            if bounding_boxes[seletedIndex][2] - bounding_boxes[i][0] > (bounding_boxes[seletedIndex][2] - bounding_boxes[seletedIndex][0])/4:
                isOverlap = True
                break

    return isOverlap


''' Check whether the face is too small
    If the detected face width and height is one-third smaller than img width and height, return true.
    return: True/False
'''
def is_too_small(bounding_box, img):
    if (bounding_box[2]-bounding_box[0]) < img.size[0]/3 and (bounding_box[3]-bounding_box[1] < img.size[1]/3):
        return True
    return False


''' Check the face is frontal or not
    return: True/False
'''
def is_frontal(bounding_box, landmark):
    if (landmark[1] - landmark[0]) < (bounding_box[2] - bounding_box[0])/3:
        return False
    return True


''' Align face to reference points & Crop it into 112*112 size
    return Image
'''
def align_to_reference_coordinate(img, landmark, reference, crop_size):
    facial5points = [[landmark[j], landmark[j + 5]] for j in range(5)]
    warped_face = warp_and_crop_face(np.array(img), facial5points, reference, (crop_size, crop_size), 'similarity')
    img_warped = Image.fromarray(warped_face)
    return img_warped


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "face alignment")
    parser.add_argument("-source_root", "--source_root", help = "specify your source dir", default = "../datasets/CASIA_WebFace", type = str)
    parser.add_argument("-dest_root", "--dest_root", help = "specify your destination dir", default = "../datasets/CASIA_WebFace_Crop_Resize", type = str)
    parser.add_argument("-crop_size", "--crop_size", help = "specify size of aligned faces, align and crop with padding", default = 112, type = int)
    args = parser.parse_args()

    source_root = args.source_root  # specify your source dir
    dest_root = args.dest_root      # specify your destination dir
    crop_size = args.crop_size      # specify size of aligned faces, align and crop with padding
    if not os.path.isdir(dest_root): 
        os.mkdir(dest_root)
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square = True) * scale
    
    ''' 這邊是 for MacOS
    cwd = os.getcwd() # delete '.DS_Store' existed in the source_root
    os.chdir(source_root)
    os.system("find . -name '*.DS_Store' -type f -delete")
    os.chdir(cwd)
    '''

    ''' 步驟:
    1. 去除 找不到 landmark 的照片
    2. 相片若有複數人臉，只取 max(b-box寬度) 的當作此張相片的人臉
    3. 去除 兩b-box重疊部分寬度 > 1/4 * 主b-box寬度 的圖片  (避免整理後還是會有偵測出兩臉情形) 
    4. 去除 b-box寬度 < 1/3 * 原圖寬度 的相片               (去除臉太小，可能偵測錯誤的情況)
    5. 去除 兩眼水平距離 < 1/3 * b-box寬度 的相片           (去除側臉)
    6. 將五官對應到一張 reference 座標點上，再擷取 112*112 大小的圖像作為新圖
    '''
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
            img = Image.open(img_path)
            
            # Step 1. If the landmarks cannot be detected, the img will be discarded
            result, bounding_boxes, landmarks = is_landmark_detected(img, img_path)
            if result == False:
                continue
            
            # Step 2. Get the main face from multi faces image
            if len(bounding_boxes) > 1:
                seletedIndex, bounding_box, landmark = get_main_face(bounding_boxes, landmarks, img_path)
                
                # Step 3. Eliminate those have overlap problem
                isOverlap = is_overlap(seletedIndex, bounding_boxes)    
                if isOverlap:
                    print("{} is discarded bcz there are faces too close!".format(img_path))
                    continue
            else:
                bounding_box = bounding_boxes[0]
                landmark = landmarks[0]
            
            # Step 4. Skip the image if the bounding-box width&height is 1/3 smaller than image width&height
            if is_too_small(bounding_box, img):
                print("{} is discarded bcz the detected face is too small!".format(img_path))
                continue
            
            # Step 5. Eliminate the image if the distance between two eyes is 1/3 times smaller than the bounding-box width
            if is_frontal(bounding_box, landmark) == False:
                print("{} is discarded bcz the face might not be frontal face!".format(img_path))
                continue
            
            # Step 6. Align face to the reference coordinate & Crop 112*112
            img_warped = align_to_reference_coordinate(img, landmark, reference, crop_size)
            

            '''
            # 新增測試 只把臉處擷取下來並resize至112*112
            bounding_box = bounding_boxes[0]
            img_warped = img.crop(list(bounding_box[0:4])).resize((112, 112))
            '''

            # Step 7. Save it
            if image_name.split('.')[-1].lower() not in ['jpg', 'jpeg']: # not from jpg
                image_name = '.'.join(image_name.split('.')[:-1]) + '.jpg'
            img_warped.save(os.path.join(dest_subfolder_path, image_name))