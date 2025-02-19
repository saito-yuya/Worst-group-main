import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
from models import model_attributes
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from utils_ours import *
from train import train, train_OP_init, train_OP,val_OP,test_OP
from loss import MiniLossComputer

def main():
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument('-d', '--dataset', choices=dataset_attributes.keys(), required=True)
    parser.add_argument('-s', '--shift_type', choices=shift_types, required=True)
    # Confounders
    parser.add_argument('-t', '--target_name')
    parser.add_argument('-c', '--confounder_names', nargs='+')
    # Resume?
    parser.add_argument('--resume', default=False, action='store_true')
    # Label shifts
    parser.add_argument('--minority_fraction', type=float)
    parser.add_argument('--imbalance_ratio', type=float)
    # Data
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--root_dir', default='../Group_test')
    # parser.add_argument('--root_dir', default='/group_bound_main')
    parser.add_argument('--reweight_groups', action='store_true', default=False)
    parser.add_argument('--augment_data', action='store_true', default=False)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    # Objective
    parser.add_argument('--robust', default=False, action='store_true')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--generalization_adjustment', default="0.0")
    parser.add_argument('--automatic_adjustment', default=False, action='store_true')
    parser.add_argument('--robust_step_size', default=0.01, type=float)
    parser.add_argument('--use_normalized_loss', default=False, action='store_true')
    parser.add_argument('--btl', default=False, action='store_true')
    parser.add_argument('--hinge', default=False, action='store_true')
    parser.add_argument('--loss', default='wCE')
    parser.add_argument('--vs_alpha', type=float, default=0.0)
    parser.add_argument('--vs_tau', type=float, default=1.0)
    parser.add_argument('--dont_set_seed', type=float, default=0)
    # Model
    parser.add_argument(
        '--model',
        choices=model_attributes.keys(),
        default='resnet50')
    parser.add_argument('--train_from_scratch', action='store_true', default=False)

    # Optimization
    parser.add_argument('--n_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--minimum_variational_weight', type=float, default=0)
    # Adding
    parser.add_argument('--max_epoch', default=10**3, type=int, metavar='N',
                    help='number of maximum epochs to run')
    parser.add_argument('--theta', type = float ,default=0.9,help='theta for Ours')
    parser.add_argument('--gamma', type=float, default=0.3,help='gamma for Ours and other method')
    parser.add_argument('--epsilon', type=float, default=0.0005,help='small margin of gamma')

    # Misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_progress', default=False, action='store_true')
    parser.add_argument('--log_dir', default='results')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int, default=50)
    parser.add_argument('--save_best', action='store_true', default=True)
    parser.add_argument('--save_last', action='store_true', default=True)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--reduce_lr_manually', type=int, default=0)

    args = parser.parse_args()
    check_args(args)

    log_dict = {"avg_loss_group":[],"avg_acc_group":[],"avg_acc'":[],"train_cls_acc":[],
                "train_worst":[],"val_avg_acc":[],"val_worst":[],"weight":[],"rt":[]} #log_dict["loss_lin"].append(loss_lin.sum().item())
    

    # BERT-specific configs copied over from run_glue.py
    # if args.model == 'bert':
    #     args.max_grad_norm = 1.0
    #     args.adam_epsilon = 1e-8
    #     args.warmup_steps = 0

    if os.path.exists(args.log_dir) and args.resume:
        resume = True
        mode = 'a'
    else:
        resume = False
        mode = 'w'

    ## Initialize logs
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, 'log.txt'), mode)
    # Record args
    log_args(args, logger)

    if args.dont_set_seed == 0:
        set_seed(args.seed)

    # Data
    # Test data for label_shift_step is not implemented yet
    test_data = None
    test_loader = None
    if args.shift_type == 'confounder':
        train_data, val_data, test_data = prepare_data(args, train=True)
    elif args.shift_type == 'label_shift_step':
        train_data, val_data = prepare_data(args, train=True)

    n0 = train_data._group_counts[0].tolist()
    n1 = train_data._group_counts[1].tolist()
    n2 = train_data._group_counts[2].tolist()
    n3 = train_data._group_counts[3].tolist()
    n_max = max(n0, n1, n2, n3)

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
    train_loader = train_data.get_loader(train=True, reweight_groups=args.reweight_groups, **loader_kwargs)
    val_loader = val_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    if test_data is not None:
        test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)

    data = {}
    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data
    n_classes = train_data.n_classes

    log_data(data, logger)

    ## Initialize model
    pretrained = not args.train_from_scratch
    if resume:
        model = torch.load(os.path.join(args.log_dir, 'last_model.pth'))
        d = train_data.input_size()[0]
    elif model_attributes[args.model]['feature_type'] in ('precomputed', 'raw_flattened'):
        assert pretrained
        # Load precomputed features
        d = train_data.input_size()[0]
        model = nn.Linear(d, n_classes)
        model.has_aux_logits = False
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'resnet34':
        model = torchvision.models.resnet34(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'wideresnet50':
        model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    # elif args.model == 'bert':
    #     assert args.dataset == 'MultiNLI'

    #     from pytorch_transformers import BertConfig, BertForSequenceClassification
    #     config_class = BertConfig
    #     model_class = BertForSequenceClassification

    #     config = config_class.from_pretrained(
    #         'bert-base-uncased',
    #         num_labels=3,
    #         finetuning_task='mnli')
    #     model = model_class.from_pretrained(
    #         'bert-base-uncased',
    #         from_tf=False,
    #         config=config)
    else:
        raise ValueError('Model not recognized.')
    
    ## Initialize model

    model_path = os.path.join(args.log_dir + "/weak_models")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    torch.save(model.state_dict(), model_path + '/check_point.pt')


    logger.flush()
    vs_alpha = args.vs_alpha
    if args.gpu_num == 0:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        torch.cuda.set_device(args.gpu_num) #使用したいGPUの番号を入れる
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_num}"
        device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
    ## Define the objective
    if args.hinge:
        assert args.dataset in ['CelebA', 'CUB']  # Only supports binary

        def hinge_loss(yhat, y):
            # The torch loss takes in three arguments so we need to split yhat
            # It also expects classes in {+1.0, -1.0} whereas by default we give them in {0, 1}
            # Furthermore, if y = 1 it expects the first input to be higher instead of the second,
            # so we need to swap yhat[:, 0] and yhat[:, 1]...
            torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction='none')
            y = (y.float() * 2.0) - 1.0
            return torch_loss(yhat[:, 1], yhat[:, 0], y)

        criterion = hinge_loss
    elif args.loss == 'cdt':
        criterion = VSLoss_4groups(n0=n0, n1=n1, n2=n2, n3=n3, mode='multiplicative', alpha=args.vs_alpha).cuda()
    elif args.loss == 'la':
        criterion = VSLoss_4groups(n0=n0, n1=n1, n2=n2, n3=n3, mode='additive', alpha=args.vs_alpha).cuda()
    elif args.loss == 'vs':
        criterion = VSLoss_4groups(n0=n0, n1=n1, n2=n2, n3=n3, mode='combined', alpha=args.vs_alpha).cuda()
    elif args.loss == 'la_tau':
        criterion = VSLoss_4groups(n0=n0, n1=n1, n2=n2, n3=n3, mode='la_tau', alpha=args.vs_alpha,
                                   tau=args.vs_tau).cuda()
    elif args.loss == 'vs_tau':
        criterion = VSLoss_4groups(n0=n0, n1=n1, n2=n2, n3=n3, mode='vs_tau', alpha=args.vs_alpha,
                                   tau=args.vs_tau).cuda()
    elif args.loss == 'wCE':
        per_cls_weights = [1.0 / np.array(train_data.n_groups)] * train_data.n_groups
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu_num)
        criterion = wCE_4groups(n0=n0, n1=n1, n2=n2, n3=n3, mode='wCE', weight = per_cls_weights).cuda()
        # criterion = wCE_4groups(n0=n0, n1=n1, n2=n2, n3=n3, mode='wCE', weight = torch.tensor([0.1,0.2,0.4,0.3]).cuda()).cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if resume:
        df = pd.read_csv(os.path.join(args.log_dir, 'test.csv'))
        epoch_offset = df.loc[len(df) - 1, 'epoch'] + 1
        logger.write(f'starting from epoch {epoch_offset}')
    else:
        epoch_offset = 0
    train_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, 'train.csv'), train_data.n_groups, mode=mode)
    val_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, 'val.csv'), train_data.n_groups, mode=mode)
    test_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, 'test.csv'), train_data.n_groups, mode=mode)

    #round Number(The number of weak-learner)
    # round_num = math.ceil(2*math.log(train_data.n_groups)/(args.gamma)**2)
    round_num = 3


    if args.loss == 'wCE':
        log_dict["weight"].append(per_cls_weights.to('cpu').detach().numpy().copy().tolist())

        train_OP_init(model, criterion, data, per_cls_weights ,logger,train_csv_logger, val_csv_logger, test_csv_logger, args,
            epoch_offset=epoch_offset)
        
        for t in range(round_num):
            model = Network_init(model,model_path + '/check_point.pt',args.gpu_num)
            logger.write(f'Round [{t}/{round_num}]:\n')

            train_group_acc,weak_model,rt = train_OP(model, criterion, data, per_cls_weights ,logger,train_csv_logger, val_csv_logger, test_csv_logger, args,
            epoch_offset=epoch_offset)

            torch.save(weak_model.state_dict(), model_path + f'/weak_model({t}).pt')

            per_cls_weights = Hedge(per_cls_weights,rt,train_data.n_groups,round_num,args.gpu_num)

            log_dict["weight"].append(per_cls_weights.to('cpu').detach().numpy().copy().tolist())
            log_dict["avg_acc_group"].append(train_group_acc)
            log_dict["rt"].append(rt)

        np.save(os.path.join(args.log_dir, 'weight.npy'), log_dict["weight"])
        np.save(os.path.join(args.log_dir, 'avg_acc_group.npy'), log_dict["avg_acc_group"])

    else:
        train(model, criterion, data, logger, train_csv_logger, val_csv_logger, test_csv_logger, args,
            epoch_offset=epoch_offset)

    if args.loss == 'wCE':
        ## val
        ht = []
        for t in tqdm(range(round_num),leave=False):
            model.load_state_dict(torch.load(model_path +f'/weak_model({t}).pt'))
            val_group_acc, val_labels_preds = val_OP(model, criterion, data, per_cls_weights ,logger,train_csv_logger, val_csv_logger, test_csv_logger, args,
                epoch_offset=epoch_offset)
            ht.append(val_labels_preds)


        calc = MiniLossComputer(dataset=data['val_data'])
        # val_group_acc,_, = calc.loss(torch.tensor(ht),data['val_data'].get_class_labels(),data['val_data'].get_group_labels())

        res = best_N(ht,data['val_data'].get_class_labels().tolist())
        ave_accuracy = ave(ht,data['val_data'].get_class_labels().tolist())
        worst,n,idx = worst_val_idx(res)
        print("val_accuracy :",ave_accuracy[n])
        print("val_worst :",worst)
        print("number of models :",n)
        print("worst_idx :",idx)

        ## test
        ht = []
        for t in tqdm(range(n)):
            model.load_state_dict(torch.load(model_path +f'/weak_model({t}).pt'))

            test_group_acc, labels_preds = test_OP(model, criterion, data, per_cls_weights ,logger,train_csv_logger, val_csv_logger, test_csv_logger, args,
                epoch_offset=epoch_offset)
            ht.append(labels_preds)
        ht = transposition(ht)
        h = voting(ht)

        calc = MiniLossComputer(dataset=data['test_data'])
        
        test_group_acc,_, = calc.loss(torch.tensor(h),data['test_data'].get_class_labels(),data['test_data'].get_group_labels())
        np.save(os.path.join(args.log_dir, 'test_group_acc.npy'), test_group_acc.to('cpu').detach().numpy().copy().tolist())

            ## Rader Charts
        label_name = [str(i+1) for i in range(test_data.n_groups)]
        plot_rader(label_name,[test_group_acc.to('cpu').detach().numpy().copy().tolist()],os.path.join(args.log_dir,'test_rader.png'))


        ## Weight
        weight_show(log_dict["weight"], train_data.n_groups ,os.path.join(args.log_dir, 'group_weight.png'))

        np.save(os.path.join(args.log_dir, 'all_preds.npy'), h)

    else : 
        gt, g,f = feature(model,train_loader,device)
        plot_feature_space(f, gt, 'class', args.log_dir)
        plot_feature_space(f, g, 'group',  args.log_dir)


    train_csv_logger.close()
    val_csv_logger.close()
    test_csv_logger.close()
    print('Finish main')


def check_args(args):
    if args.shift_type == 'confounder':
        assert args.confounder_names
        assert args.target_name
    elif args.shift_type.startswith('label_shift'):
        assert args.minority_fraction
        assert args.imbalance_ratio


class VSLoss_4groups(nn.Module):
    def __init__(self, n0, n1, n2, n3, alpha, tau=1, mode='multiplicative', weight=None):
        super(VSLoss_4groups, self).__init__()
        n_max = max(n0, n1, n2, n3)
        n_sum = sum([n0, n1, n2, n3])
        if mode == 'multiplicative':
            iota_list = np.zeros(4)
            Delta_list = np.array([n0 / n_max, n1 / n_max, n2 / n_max, n3 / n_max]) ** alpha
        elif mode == 'additive':
            iota_list = np.array([n0 / n_max, n1 / n_max, n2 / n_max, n3 / n_max]) ** -alpha
            Delta_list = np.ones(4)
        elif mode == 'combined':
            iota_list = np.array([n0 / n_max, n1 / n_max, n2 / n_max, n3 / n_max]) ** -alpha
            Delta_list = np.array([n0 / n_max, n1 / n_max, n2 / n_max, n3 / n_max]) ** alpha
        elif mode == 'la_tau':
            iota_list = -tau * np.array(
                [np.log(n0 / n_sum), np.log(n1 / n_sum), np.log(n2 / n_sum), np.log(n3 / n_sum)])
            Delta_list = np.ones(4)
        elif mode == 'vs_tau':
            iota_list = -tau * np.array(
                [np.log(n0 / n_sum), np.log(n1 / n_sum), np.log(n2 / n_sum), np.log(n3 / n_sum)])
            Delta_list = np.array([n0 / n_max, n1 / n_max, n2 / n_max, n3 / n_max]) ** alpha

        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.Delta_list = torch.cuda.FloatTensor(Delta_list)
        self.weight = weight

    def forward(self, x, target, group):
        group = group.squeeze(-1)
        index = torch.zeros((x.shape[0], 2), dtype=torch.uint8)
        group_one_hot = torch.nn.functional.one_hot(group, num_classes=4)
        batch_iota = torch.matmul(group_one_hot.float(), self.iota_list.view(4, 1).repeat(1, 2).float())
        batch_Delta = torch.matmul(group_one_hot.float(), self.Delta_list.view(4, 1).repeat(1, 2).float())

        output = x * batch_Delta - batch_iota
        cel = torch.nn.CrossEntropyLoss(reduction='none')
        return cel(output, target)

class wCE_4groups(nn.Module):
    def __init__(self, n0, n1, n2, n3, mode='wCE', weight=None):
        super(wCE_4groups, self).__init__()
        n_max = max(n0, n1, n2, n3)
        n_sum = sum([n0, n1, n2, n3])

        if mode == 'wCE': ## group weight CE
            self.mode = mode

        elif mode == 'OCO':
            print('OCO')
        else:
            print(mode)

        self.weight = weight

    def forward(self, outputs, target, group):
        group = group.squeeze(-1)
        # index = torch.zeros((outputs.shape[0], 2), dtype=torch.uint8)
        group_one_hot = torch.nn.functional.one_hot(group, num_classes=4).float()

        if self.weight is not None:
            weights = torch.matmul(group_one_hot, self.weight.view(-1, 1)).squeeze() 
        else:
            weights = torch.ones(group.size(0)).cuda()  # 重みがない場合は全て1
        cel = nn.CrossEntropyLoss(reduction='none')
        loss = cel(outputs, target)

        weighted_loss = loss * weights

        return weighted_loss

if __name__ == '__main__':
    main()
