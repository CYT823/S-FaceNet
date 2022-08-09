import sys
import os
sys.path.append(os.path.join(sys.path[0], 'MTCNN'))
from MTCNN import create_mtcnn_net
from utils.align_trans import *
import numpy as np
from torchvision import transforms as trans
import torch
from face_model import MobileFaceNet, l2_norm
from pathlib import Path
import cv2
import argparse

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def prepare_facebank(model, path = 'facebank', tta = True):
    embeddings = []
    names = ['Unknown']
    data_path = Path(path)

    for doc in data_path.iterdir():

        if doc.is_file():
            continue
        else:
            embs = []
            for files in listdir_nohidden(doc):
                image_path = os.path.join(doc, files)
                img = cv2.imread(image_path)

                if img.shape != (112, 112, 3):
                    bboxes, landmarks = create_mtcnn_net(img, 20, device,
                                                     p_model_path='MTCNN/weights/pnet_Weights',
                                                     r_model_path='MTCNN/weights/rnet_Weights',
                                                     o_model_path='MTCNN/weights/onet_Weights')

                    img = Face_alignment(img, default_square=True, landmarks=landmarks)

                with torch.no_grad():
                    if tta:
                        mirror = cv2.flip(img, 1)
                        emb = model(test_transform(img).to(device).unsqueeze(0))
                        emb_mirror = model(test_transform(mirror).to(device).unsqueeze(0))
                        embs.append(l2_norm(emb + emb_mirror))
                    else:
                        embs.append(model(test_transform(img).to(device).unsqueeze(0)))

            if len(embs) == 0:
                continue
            embedding = torch.cat(embs).mean(0, keepdim=True)
            embeddings.append(embedding)
            names.append(doc.name)

    embeddings = torch.cat(embeddings)
    names = np.array(names)
    print(embeddings)
    print(names)
    torch.save(embeddings, os.path.join(path, 'facebank.pth'))
    np.save(os.path.join(path, 'names'), names)

    return embeddings, names

def load_facebank(path = 'facebank'):
    data_path = Path(path)
    embeddings = torch.load(data_path/'facebank.pth')
    names = np.load(data_path/'names.npy')
    return embeddings, names

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Facebank file creating")
    parser.add_argument("-i", "--iterations", help="Which weight you wanna use", default=3000, type=int)
    args = parser.parse_args()

    test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # embeding size is 256 (feature vector)
    detect_model = MobileFaceNet(256).to(device)
    model_checkpoint = torch.load('saving_CASIA_ckpt_0526/Iter_{:0>8d}_model.ckpt'.format(args.iterations), map_location=lambda storage, loc: storage)
    detect_model.load_state_dict(model_checkpoint['net_state_dict'], strict=False) 
    detect_model.to(device)
    print('MobileFaceNet face detection model generated.')  

    detect_model.eval()

    embeddings, names = prepare_facebank(detect_model, path = 'facebank', tta = True)
    print(embeddings.shape)






