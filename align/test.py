from PIL import Image
from detector import detect_faces
from visualization_utils import show_results
import argparse

'''用來單張測試偵測結果'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "face detection test")
    parser.add_argument("-s", "--source_root", help = "specify your source dir", default = "../datasets/CASIA_WebFace/0000168/276.jpg", type = str)
    args = parser.parse_args()
    source_root = args.source_root

    img = Image.open(source_root) # modify the image path to yours
    bounding_boxes, landmarks = detect_faces(img) # detect bboxes and landmarks for all faces in the image
    print(bounding_boxes)
    print(landmarks)
    temp = show_results(img, bounding_boxes, landmarks) # visualize the results
    temp.show()