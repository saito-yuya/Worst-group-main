import sys
import os
import torch
import numpy as np
import csv
import math
from collections import defaultdict
import time
from os import TMP_MAX
from torch import nn
import matplotlib.pyplot as plt
import torch.optim as optim
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from itertools import chain 
import sklearn.metrics as metrics
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils.multiclass import unique_labels
from sklearn.manifold import TSNE
import torch.nn as nn

class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class CSVBatchLogger:
    def __init__(self, csv_path, n_groups, mode='w'):
        columns = ['epoch', 'batch']
        for idx in range(n_groups):
            columns.append(f'avg_loss_group:{idx}')
            columns.append(f'exp_avg_loss_group:{idx}')
            columns.append(f'avg_acc_group:{idx}')
            columns.append(f'processed_data_count_group:{idx}')
            columns.append(f'update_data_count_group:{idx}')
            columns.append(f'update_batch_count_group:{idx}')
        columns.append('avg_actual_loss')
        columns.append('avg_per_sample_loss')
        columns.append('avg_acc')
        columns.append('model_norm_sq')
        columns.append('reg_loss')

        self.path = csv_path
        self.file = open(csv_path, mode)
        self.columns = columns
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if mode=='w':
            self.writer.writeheader()

    def log(self, epoch, batch, stats_dict):
        stats_dict['epoch'] = epoch
        stats_dict['batch'] = batch
        self.writer.writerow(stats_dict)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_args(args, logger):
    for argname, argval in vars(args).items():
        logger.write(f'{argname.replace("_"," ").capitalize()}: {argval}\n')
    logger.write('\n')


def make_label_clsnum(loader):
    label = []
    cls_num = []
    for i in loader:
        _,labels = i
        label.extend(labels.tolist())
    n = len(set(label))
    for i in range(n):
        cls_num.append(label.count(i))
        
    return label,cls_num

# def torch_fix_seed(seed=0):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.use_deterministic_algorithms = True
#     os.environ['PYTHONHASHSEED'] = str(seed)

def prepare_folders(args):
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


class Feedbuck():
    def ACC(TP,num):
        return TP/num

    def Binary_ACC(accuracy,theta):
        if (accuracy) >= theta:
            return 1
        else:
            return 0
        
feed = Feedbuck.Binary_ACC
    
def Network_init(model,path,device):
    model.load_state_dict(torch.load(path))
    model = model.cuda(device)
    return model

def OP_init(model,train_loader,optimizer,criterion,device):
    for data in train_loader:
        inputs, labels = data
        labels = labels.reshape(-1)
        inputs = inputs.cuda(device)
        labels = labels.cuda(device)
        optimizer.zero_grad()
        outputs,_ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print("OP_init Finish")

def class_wise_acc(model,loader,device):
    class_acc_list,y_preds,true_label = [],[],[]
    model = model.cuda(device)
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate((loader)):
                inputs, labels = data
                labels = labels.reshape(-1)
                inputs = inputs.cuda(device)
                labels = labels.cuda(device)
                predicted,_ = model(inputs) 
                predicted = torch.max(predicted, 1)[1]
                y_preds.extend(predicted.cpu().numpy())
                true_label.extend(labels.cpu().numpy())
        cf = confusion_matrix(true_label,y_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = np.divide(cls_hit, cls_cnt, out=np.zeros_like(cls_hit), where=cls_cnt !=0)
        # cls_acc = cls_hit / cls_cnt
        class_acc_list.append(cls_acc)
    model.train()
    return class_acc_list[0],y_preds,true_label,np.round(confusion_matrix(true_label,y_preds,normalize='true'),2)


def calc_acc(label,pred):
    class_acc_list = []
    cf = confusion_matrix(label,pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = np.divide(cls_hit, cls_cnt, out=np.zeros_like(cls_hit), where=cls_cnt !=0)
    cls_acc = np.around(cls_acc ,decimals=4)
    # cls_acc = cls_hit / cls_cnt
    class_acc_list.append(cls_acc.tolist())
    return class_acc_list

def class_wise_acc_h(y_pred,labels):
    ans = defaultdict(int)
    for item in zip(labels,y_pred):
        if item[0] == item[1]:
            ans[item[0]] += 1
    return ans 

def train(model,train_loader,classes,weight_tmp,optimizer,criterion,max_epoch,theta,gamma,log,device):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.train()
    count = 0
    train_acc_list = []

    end = time.time()

    for epoch in range(max_epoch):
        running_loss = 0.0
        criterion = nn.CrossEntropyLoss(weight=weight_tmp).cuda(device)
        for images, labels in tqdm(train_loader, leave=False):
            labels = labels.reshape(-1)
            images, labels = images.cuda(device), labels.cuda(device)
            outputs,_ = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss/len(train_loader)
        # epoch_list.append(epoch)
        
        train_acc_list,y_preds,_,_ = class_wise_acc(model,train_loader,device)

        print("-----------------total_epoch:{}------------------".format(epoch))
        print("train_loss:{}".format(train_loss))

        weight_l = weight_tmp.tolist()

        rt = [0]*classes
        for k in range(classes):
            rt[k] = feed(train_acc_list[k],theta)
        # r.append(rt)

        ft = 0
        for k in range(classes):
            ft +=  weight_l[k]*rt[k]

        # DNN のearly_stoppingの条件
        if ft >=  (1/2) + gamma:
            # f.append(np.dot(weight_l,rt))
            print("Satisfied with W.L Definition : {}".format(epoch))
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            batch_time.update(time.time() - end)
            output = (
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss:.4f} ({loss:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                batch_time=batch_time,loss=train_loss, top1=top1, top5=top5))
            print(output)
            log.write(output + '\n')
            end = time.time()
            break
        elif epoch == (max_epoch-1):
            print("Couldn't Satisfied with W.L Definition")
            sys.exit()

    return train_acc_list,model,y_preds,rt

def Hedge(weight_tmp,rt,classes,round_num,device):
    eta = ((8*(math.log(classes)))/(round_num))**(1/2)
    down = 0
    for i in range(classes):
        down +=  (weight_tmp[i].item()*math.exp(-(eta*rt[i])))
    for i,item in enumerate(rt):
        weight = weight_tmp[i].item()
        weight_tmp[i] = ((weight*math.exp(-(eta*item)))/down)

    weight_tmp = torch.tensor(weight_tmp)
    weight_tmp = weight_tmp.cuda(device)
    return weight_tmp

def transposition(matrix):
    matrix = np.array(matrix).T
    matrix = matrix.tolist()
    return matrix

#input:各モデル（弱学習器）の予測ラベル
#output:多数決後（強学習器）の予測ラベル
def voting(ht):
    h_var = []
    # ht = treatment(ht)
    for m in range(len(ht)):
        count = Counter((ht[m]))
        majority = count.most_common()
        h_var.append(majority[0][0])
    h_var = transposition(h_var)
    return h_var

# votingの入力が違うversion
def ensemble(ht,keep):
    keep.append(ht)
    #  ht = treatment(ht)
    keep = transposition(keep)
    h_var = []
    for m in range(len(keep)):
        count = Counter((keep[m]))
        majority = count.most_common()
        h_var.append(majority[0][0])
    h_var = transposition(h_var)
    return h_var

# inp : model_i　における予測list,正解ラベル
# out : model_iまでの予測多数決のaccuracy list
def best_N(out_list,y_true):
    keep = []
    acc_list = []
    for i in range(len(out_list)):
        out = out_list[i]
        if i != 0:
            res = ensemble(out,keep)
        else:
            res = out
        acc = calc_acc(y_true,res)
        keep.append(out)
        acc_list.append(acc[0])
    return acc_list

# out_list = [[0,1,2],[1,1,1],[1,1,1]]
# y_true = [0,1,2]
# print(best_N(out_list,y_true))

# def confusion_matrix_show(labels_l, predictions_l,flag,path):
#     cm = confusion_matrix(labels_l, predictions_l)

#     print(cm)
#     sns.heatmap(cm,square=True, cbar=True, annot=True, cmap='Reds',fmt='.5g')
#     plt.yticks(rotation=0)
#     plt.xlabel("Pre", fontsize=13, rotation=0)
#     plt.ylabel("GT", fontsize=13)
#     if flag == 0:
#         plt.savefig(path + 'theta,gamma='+str(theta)+','+str(gamma)+''+'_train.pdf')
#     else:
#         plt.savefig(path + 'theta,gamma='+str(theta)+','+str(gamma)+''+'_test.pdf')
#     plt.close()

def worst_val_idx(acc_list):
    acc_list = np.array(acc_list)
    idx = acc_list.argmin(axis=1)
    val = acc_list.min(axis=1)
    # n = val.argmax()
    n = max([i for i,x in enumerate(val) if x == max(val)])
    worst = val[n]
    return worst,n,idx

# Input : Weight 1*x 
def weight_show(weight,classes,path):

    weight_s = transposition(weight)

    sns.set()
    sns.set_style(style='whitegrid')
    sns.set_palette("husl",classes)
    
    # 目盛を内側にする。
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    fig = plt.figure()
    ax1 = plt.subplot(111)
    
    # グラフの上下左右に目盛線を付ける。
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    
    # 軸のラベルを設定する。
    ax1.set_xlabel('Round t')
    ax1.set_ylabel('Weight') 

    #データ点のプロット結果
    for i in range(len(weight[0])):
        plt.plot(np.arange(len(weight_s[0])),  weight_s[i], 'o-', lw =1, label = "class : {}".format(i+1))

    ax1.legend(loc = "lower left", fontsize = 10)
    
    plt.savefig(path,dpi=400)
    plt.close()

    return True

def calc_acc_ave(label,pred):
    # print(pred)
    ave_list = []
    class_wise_list = []
    cf = confusion_matrix(label,pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    ## Adding
    cls_acc = np.divide(cls_hit, cls_cnt, out=np.zeros_like(cls_hit), where=cls_cnt !=0)
    cls_acc = np.around(cls_acc ,decimals=8)
    ave = sum(cls_hit)/sum(cls_cnt)

    ave_list.append(ave.tolist())
    class_wise_list.append(cls_acc.tolist())
    return ave_list,cls_acc

def ave(out_list,y_true):
    keep = []
    ave_list = []
    cls_acc_list = []
    for i in range(len(out_list)):
        out = out_list[i]
        if i != 0:
            res = ensemble(out,keep)
            # print(res)
        else:
            res = out
        acc,cls_wise_acc = calc_acc_ave(y_true,res)
        keep.append(out)
        ave_list.append(acc[0])
        cls_acc_list.append(cls_wise_acc.tolist())
    # return ave_list,cls_acc_list
    return ave_list

# def cls_acc_ens(,y_pred):
#     keep = []
#     ave_list = []
#     for i in range(len(out_list)):
#         out = out_list[i]
#         if i != 0:
#             res = ensemble(out,keep)
#             # print(res)
#         else:
#             res = out
#         acc = calc_acc_ave(y_true,res)
#         keep.append(out)
#         ave_list.append(acc[0])
#     return ave_list

def save_confusion_matrix(cm, path, title=''):
    plt.figure()
    # cm = cm / cm.sum(axis=-1, keepdims=1)
    sns.heatmap(cm, annot=True, cmap='Blues_r', fmt='.2f')
    plt.xlabel('pred')
    plt.ylabel('GT')
    plt.title(title)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def plot_rader(labels, values ,save_path):
    
    #描画領域の作成
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={'projection': 'polar'})
    
    colors = [ 'y', 'b', 'r', 'g', 'm']
    
    #チャートを順に描画
    for i, data in enumerate(zip(values, colors)):        
        d = data[0]
        color = data[1]
        
        #要素数の連番作成
        angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)
        #閉じた多角形に変換
        value = np.concatenate((d, [d[0]]))  
        
        #線の描画
        ax.plot(angles, value, 'o-', color=color)
        #塗りつぶし
        ax.fill(angles, value, alpha=0.25, facecolor=color)  
        ax.set_rlim(0, 1.0)

    #軸ラベルの設定
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)  
    #描画開始位置の指定（N: North）
    ax.set_theta_zero_location('N')

    plt.title(save_path)
    
    plt.savefig(save_path,dpi=400)
    plt.close(fig)
    

def feature(model,loader,device):
    labels ,groups,features = [],[],[]
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(loader):
            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            f = feature_extractor(x).squeeze()
            features.extend(f.cpu().detach().numpy())
            labels.extend(y.cpu().detach().numpy())
            groups.extend(g.cpu().detach().numpy())
    model.train()
    return np.array(labels),np.array(groups),np.array(features)


def plot_feature_space(features, labels, mode, path): #mode = {group/class} 
    num_classes = max(labels)+1
    num_samples = min(labels.shape[0], 5000) 
    show_index = np.random.choice(np.arange(labels.shape[0]), num_samples, replace=False)
    features_2d = TSNE(n_components=2, random_state=0).fit_transform(features[show_index])

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(num_classes):
        target = features_2d[labels[show_index] == i]
        if mode == 'group':
            ax.scatter(x=target[:, 0], y=target[:, 1], label=f'group {i}', alpha=0.3, color=f'C{i}')
        else : ## mode == 'class':
            ax.scatter(x=target[:, 0], y=target[:, 1], label=f'class {i}', alpha=0.3, color=f'C{i}')

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=20, markerscale=2)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    plt.savefig(f'{path}/feature_space_{mode}.png', bbox_inches='tight', dpi=400)
    plt.close()

def weighted_accuracy(class_counts, class_accuracies):
    total_samples = sum(class_counts)
    weighted_acc = sum(c * a for c, a in zip(class_counts, class_accuracies)) / total_samples
    return weighted_acc
