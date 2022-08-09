"""
Evaluation of LFW, CFP-FP and AgeDB
"""
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from face_model import MobileFaceNet, l2_norm
from datasets.dataloader import LFW, CFP_FP, AgeDB
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getWrongPrediction(scores, flags, threshold, nameLs, nameRs):
    sum = 0
    for i in range(0, len(flags)):
        if(flags[i] == 1 and scores[i] > threshold):
            print(nameLs[i], nameRs[i], flags[i])
            sum += 1
        elif(flags[i] == -1 and scores[i] < threshold):
            print(nameLs[i], nameRs[i], flags[i])
            sum += 1
    print('###', sum, '###')

def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] < threshold)
    n = np.sum(scores[flags == -1] > threshold)
    return 1.0 * (p + n) / len(scores)

def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 3.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])
    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold

def getFeature(net, dataloader, device, flip = True):
### Calculate the features ###
    featureLs = None
    featureRs = None 
    count = 0
    for det in dataloader:
        for i in range(len(det)):
            det[i] = det[i].to(device)
        count += det[0].size(0)
#        print('extracing deep features from the face pair {}...'.format(count))
    
        with torch.no_grad():
            res = [net(d).data.cpu() for d in det]
            
        if flip:      
            featureL = l2_norm(res[0] + res[1])
            featureR = l2_norm(res[2] + res[3])
        else:
            featureL = res[0]
            featureR = res[2]
        
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = torch.cat((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = torch.cat((featureRs, featureR), 0)
        
    return featureLs, featureRs

def evaluation_10_fold(featureL, featureR, dataset, method = 'l2_distance'):
    
    ### Evaluate the accuracy ###
    ACCs = np.zeros(10)
    threshold = np.zeros(10)
    fold = np.array(dataset.folds).reshape(1,-1)
    flags = np.array(dataset.flags).reshape(1,-1)
    # nameLs = np.array(dataset.nameLs)
    # nameRs = np.array(dataset.nameRs)
    featureLs = featureL.numpy()
    featureRs = featureR.numpy()

    for i in range(10):
        
        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)
        
        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        if method == 'l2_distance':
            scores = np.sum(np.power((featureLs - featureRs), 2), 1) # L2 distance
        elif method == 'cos_distance':
            scores = np.sum(np.multiply(featureLs, featureRs), 1) # cos distance

        threshold[i] = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold[i])

        # 如果要印出錯誤辨識的結果，再開啟下面這行，記得開啟 77、78 行的 nameLs 和 nameRs
        # getWrongPrediction(scores[testFold[0]], flags[testFold[0]], threshold[i], nameLs[testFold[0]], nameRs[testFold[0]])
        
    return ACCs, threshold

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face_Detection_Evaluation')
    parser.add_argument('--dataset', type=str, default='LFW', help='Select the dataset to evaluate, LFW, CFP-FP, AgeDB')
    parser.add_argument('--method', type=str, default='l2_distance', help='methold to calculate feature similarity, l2_distance, cos_distance')
    parser.add_argument('--flip', type=str, default=True, help='if flip the image with time augmentation')
    parser.add_argument('--iteration', '-i', type=int, default=None, help='load checkpoint')
    parser.add_argument('--modelNum', '-n', type=str, default=None, help='load checkpoint')
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  

    # 載入MobileFaceNet參數
    model = MobileFaceNet(256)
    model_checkpoint = torch.load('saving_CASIA_ckpt_{}/Iter_{:0>8d}_model.ckpt'.format(args.modelNum, args.iteration)) 
    model.load_state_dict(model_checkpoint['net_state_dict'])   
    model.to(device)
    print('MobileFaceNet has been loaded!')

    model.eval()

    ### load data ###
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  # range [0.0, 1.0] -> [-1.0,1.0]
    
    select_dataset = args.dataset
    
    if select_dataset == 'LFW':
        root = 'datasets/LFW_Aligned'
        file_list = 'datasets/LFW_Aligned/pairs.txt'
        dataset = LFW(root, file_list, transform=transform)
    elif select_dataset == 'CFP-FP':
        root = 'datasets/CFP-FP_Aligned'
        file_list = 'datasets/CFP-FP_Aligned/pairs.txt'
        dataset = CFP_FP(root, file_list, transform=transform)
    elif select_dataset == 'AgeDB':
        root = 'datasets/AgeDB_Aligned'
        file_list = 'datasets/AgeDB_Aligned/01_pairs.txt'
        dataset = AgeDB(root, file_list, transform=transform)    
    
    dataloader = data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, drop_last=False)
    print('{} data has loaded with length'.format(select_dataset), len(dataset))
    
    featureLs, featureRs = getFeature(model, dataloader, device, flip = args.flip)
    ACCs, threshold = evaluation_10_fold(featureLs, featureRs, dataset, method = args.method)
    
    for i in range(len(ACCs)):
        print('{} accuracy: {:.2f} threshold: {:.4f}'.format(i+1, ACCs[i] * 100, threshold[i]))

    print('\n------------------------\n')
    print('saving_CASIA_ckpt_{}/Iter_{:0>8d}_model.ckpt'.format(args.modelNum, args.iteration))
    print('{} Average acc:{:.4f} average threshold:{:.4f}'.format(select_dataset, np.mean(ACCs) * 100, np.mean(threshold)))