import os 
import sys
sys.path.append('..')
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim
from torch.optim import lr_scheduler
from datasets.dataloader import LFW, CFP_FP, AgeDB, CASIAWebFace, MS1M, VGGFace
from face_model import MobileFaceNet, Arcface
import time
from Evaluation import getFeature, evaluation_10_fold

def load_data(batch_size, dataset = 'Faces_emore'):
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  # range [0.0, 1.0] -> [-1.0,1.0]
    
    root = 'datasets/LFW_Aligned'
    file_list = 'datasets/LFW_Aligned/pairs.txt'
    dataset_LFW = LFW(root, file_list, transform=transform)
    
    root = 'datasets/CFP-FP_Aligned'
    file_list = 'datasets/CFP-FP_Aligned/pairs.txt'
    dataset_CFP_FP = CFP_FP(root, file_list, transform=transform)
        
    root = 'datasets/AgeDB_Aligned'
    file_list = 'datasets/AgeDB_Aligned/01_pairs.txt'
    dataset_AgeDB = AgeDB(root, file_list, transform=transform) 
    
    if dataset == 'CASIA':
        root = 'datasets/CASIA_WebFace_Aligned'
        file_list = 'datasets/CASIA_WebFace_Aligned/webface_align_112_start_from_0.txt'
        # root = 'datasets/CASIA_WebFace_Aligned_Augmentation'
        # file_list = 'datasets/CASIA_WebFace_Aligned_Augmentation/webface_align_112.txt'
        dataset_train = CASIAWebFace(root, file_list, transform=transform)
    elif dataset == 'VGG_Face2':
        # root = 'datasets/VGG_Face2_Resize'
        # file_list = 'datasets/VGG_Face2_Resize/vgg_face2_resize_112.txt'
        root = 'datasets/VGG_Face2_Aligned'
        file_list = 'datasets/VGG_Face2_Aligned/vgg_face2_112.txt'
        dataset_train = VGGFace(root, file_list, transform=transform) 
    elif dataset == 'Faces_emore':
        root = 'datasets/faces_emore_images'
        file_list = 'datasets/faces_emore_images/faces_emore_align_112.txt'
        dataset_train = MS1M(root, file_list, transform=transform) 
    else:
        raise NameError('no training data exist!')
    
    dataloaders = {'train': data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2),
                   'LFW': data.DataLoader(dataset_LFW, batch_size=batch_size, shuffle=False, num_workers=2),
                   'CFP_FP': data.DataLoader(dataset_CFP_FP, batch_size=batch_size, shuffle=False, num_workers=2),
                   'AgeDB': data.DataLoader(dataset_AgeDB, batch_size=batch_size, shuffle=False, num_workers=2)}  

    dataset = {'train': dataset_train,'LFW': dataset_LFW, 
               'CFP_FP': dataset_CFP_FP, 'AgeDB': dataset_AgeDB}

    dataset_sizes = {'train': len(dataset_train), 'LFW': len(dataset_LFW),
                     'CFP_FP': len(dataset_CFP_FP), 'AgeDB': len(dataset_AgeDB)}

    return dataloaders, dataset_sizes, dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face_Detection_Training')
    parser.add_argument('--dataset', type=str, default='CASIA', help='Training dataset: CASIA, VGG_Face2, Faces_emore')
    parser.add_argument('--feature_dim', type=int, default=256, help='the feature dimension output')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training and evaluation')
    parser.add_argument('--epoch', type=int, default=10, help='number of epoches for training')
    parser.add_argument('--method', type=str, default='l2_distance', help='methold to evaluate feature similarity, l2_distance, cos_distance')
    parser.add_argument('--flip', type=str, default=True, help='if flip the image with time augmentation')
    parser.add_argument('--start_iter', type=int, default=None, help='load checkpoint')
    args = parser.parse_args()

    # load data
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
    dataloaders , dataset_sizes, dataset = load_data(args.batch_size, dataset = args.dataset)
    print('training and validation data have been loaded!')

    # load model
    model = MobileFaceNet(args.feature_dim) # embedding size
    margin = Arcface(embedding_size=args.feature_dim, classnum=int(dataset['train'].class_nums), s=32., m=0.5)

    if args.start_iter != None and os.path.isdir('saving_{}_ckpt'.format(args.dataset)):
        model_checkpoint = torch.load('saving_{}_ckpt/Iter_{:0>8d}_model.ckpt'.format(args.dataset, args.start_iter))
        model.load_state_dict(model_checkpoint['net_state_dict'])   # 載入MobileFaceNet參數

        margin_checkpoint = torch.load('saving_{}_ckpt/Iter_{:0>8d}_model.ckpt'.format(args.dataset, args.start_iter))
        margin.load_state_dict(margin_checkpoint['net_state_dict'], strict=False) # 載入ArcFace參數

    model.to(device)  
    margin.to(device)
    print('MobileFaceNet & ArcFace have been loaded!')

    # load optimizer & learning_rate_scheduler
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD(
                    [{'params': model.parameters(), 'weight_decay': 5e-4},
                        {'params': margin.parameters(), 'weight_decay': 5e-4}], 
                            lr=0.01, momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[6, 8, 10], gamma=0.3) 
    
    # create log file and folder
    train_logging_file = 'train_{}_logging.txt'.format(args.dataset)
    test_logging_file = 'test_{}_logging.txt'.format(args.dataset)
    save_dir = 'saving_{}_ckpt'.format(args.dataset)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    print("{} folder has been created!".format(save_dir))

    # record parameter
    best_acc = {'LFW': 0.0, 'CFP_FP': 0.0, 'AgeDB': 0.0}
    best_iters = {'LFW': 0, 'CFP_FP': 0, 'AgeDB': 0}
    total_iters = 0 if args.start_iter==None else int(args.start_iter)

    start = time.time()
    print('training kicked off from iteration {}'.format(0 if args.start_iter==None else int(args.start_iter)))
    print('=' * 40) 

    for epoch in range(args.epoch):
        model.train()     
        since = time.time()
        for det in dataloaders['train']: 
            img, label = det[0].to(device), det[1].to(device)
            optimizer_ft.zero_grad()
            # train model
            with torch.set_grad_enabled(True):
                raw_logits = model(img)
                output = margin(raw_logits, label)

                loss = criterion(output, label)
                loss.backward()
                optimizer_ft.step()
                
                total_iters += 1
                if total_iters % 100 == 0:
                    _, preds = torch.max(output.data, 1)

                    total = label.size(0)
                    correct = (np.array(preds.cpu()) == np.array(label.data.cpu())).sum()                  
                    time_cur = (time.time() - since) / 100
                    since = time.time()
                    
                    for p in  optimizer_ft.param_groups:
                        lr = p['lr']
                        
                    print("Epoch {}/{}, Iters: {:0>8d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                          .format(epoch, args.epoch-1, total_iters, loss.item(), correct/total, time_cur, lr))
                    with open(train_logging_file, 'a') as f:
                        f.write("Epoch {}/{}, Iters: {:0>8d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                          .format(epoch, args.epoch-1, total_iters, loss.item(), correct/total, time_cur, lr)+'\n')
                    f.close()

            # save model
            if total_iters % 3000 == 0:
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': model.state_dict()},
                    os.path.join(save_dir, 'Iter_%08d_model.ckpt' % total_iters))
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': margin.state_dict()},
                    os.path.join(save_dir, 'Iter_%08d_margin.ckpt' % total_iters))
            
            # evaluate accuracy
            if total_iters % 3000 == 0:
                model.eval()
                sum = 0.0 # for saving .pth file
                for phase in ['LFW', 'CFP_FP', 'AgeDB']:                 
                    featureLs, featureRs = getFeature(model, dataloaders[phase], device, flip = args.flip)
                    ACCs, threshold = evaluation_10_fold(featureLs, featureRs, dataset[phase], method = args.method)
                    print('Epoch {}/{}，{} average acc:{:.4f} average threshold:{:.4f}'
                          .format(epoch, args.epoch-1, phase, np.mean(ACCs) * 100, np.mean(threshold)))
                    if best_acc[phase] <= np.mean(ACCs) * 100:
                        best_acc[phase] = np.mean(ACCs) * 100
                        best_iters[phase] = total_iters
                    
                    with open(test_logging_file, 'a') as f:
                        f.write('Epoch {}/{}， {} average acc:{:.4f} average threshold:{:.4f}'
                                .format(epoch, args.epoch-1, phase, np.mean(ACCs) * 100, np.mean(threshold))+'\n')
                    
                    sum += np.mean(ACCs) * 100
                
                # Save .pth file if the sum of ACCs is larger than the best
                if sum > best_acc['LFW'] + best_acc['CFP_FP'] + best_acc['AgeDB'] - 5:
                    torch.save(model, os.path.join(save_dir, 'Iter_%08d_model.pth' % total_iters))
                
                model.train()
            
        exp_lr_scheduler.step()
        time_elapsed = time.time() - start 
        print('It has gone through: {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60))
    time_elapsed = time.time() - start  
    print('Finally Best Accuracy: LFW: {:.4f} in iters: {}, CFP_FP: {:.4f} in iters: {} and AgeDB: {:.4f} in iters: {}'.format(best_acc['LFW'], best_iters['LFW'], best_acc['CFP_FP'], best_iters['CFP_FP'], best_acc['AgeDB'], best_iters['AgeDB']))
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60))