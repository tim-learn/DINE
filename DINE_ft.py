import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (11 + gamma * iter_num / max_iter) ** (-power)
    # decay = (1 + gamma) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)
    dsets["target_te"] = ImageList(txt_tar, transform=image_test())
    dset_loaders["target_te"] = DataLoader(dsets["target_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item() / np.log(all_label.size()[0])

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label).astype(int),:]
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc, predict, mean_ent
    else:
        return accuracy*100, mean_ent, predict, mean_ent

def train_target(args):
    dset_loaders = data_load(args)
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
        
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
   
    modelpath = args.output_dir + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))

    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.eval()
    netB.eval()
    netC.eval()
    acc_s_te, _, pry, mean_ent = cal_acc(dset_loaders['test'], netF, netB, netC, False)
    log_str = 'Task: {}, Iter:{}/{}; Accuracy={:.2f}%, Ent={:.3f}'.format(args.name, iter_num, max_iter, acc_s_te, mean_ent)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')
    netF.train()
    netB.train()
    netC.train()

    old_pry = 0
    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter, power=0.75)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        softmax_out = nn.Softmax(dim=1)(outputs_test)
        entropy_loss = torch.mean(loss.Entropy(softmax_out))

        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = -torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        entropy_loss -= gentropy_loss
        entropy_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            acc_s_te, _, pry, mean_ent = cal_acc(dset_loaders['test'], netF, netB, netC, False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy={:.2f}%, Ent={:.3f}'.format(args.name, iter_num, max_iter, acc_s_te, mean_ent)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netF.train()
            netB.train()
            netC.train()

            if torch.abs(pry - old_pry).sum() == 0:
                break
            else:
                old_pry = pry.clone()
 
    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DINE')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=30, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'image-clef', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet18, resnet50, resnext50")
    parser.add_argument('--net_src', type=str, default='resnet50', help="alexnet, vgg16, resnet18, resnet34, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    
    args = parser.parse_args()
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = './data/'
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        args.output_dir = osp.join(args.output, args.net_src + '_' + args.net, str(args.seed), args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)
        
        args.out_file = open(osp.join(args.output_dir, 'log_finetune.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()

        train_target(args)