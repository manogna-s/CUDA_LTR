from __future__ import print_function

import argparse, os, shutil, time, random, math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import losses

from datasets.cifar100 import *

from train.train import *
from train.validate import *

from models.net import *
from losses.loss import *

from utils.config import *
from utils.plot import *
from utils.common import make_imb_data, save_checkpoint, hms_string

from utils.logger import logger

args = parse_args()
reproducibility(args.seed)
args = dataset_argument(args)
args.logger = logger(args)

best_acc = 0 # best test accuracy


def main():
    global best_acc

    try:
        assert args.num_max <= 50000. / args.num_class
    except AssertionError:
        args.num_max = int(50000 / args.num_class)
    
    print(f'==> Preparing imbalanced CIFAR-100')
    # N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, args.num_class, args.imb_ratio)
    trainset, testset = get_cifar100(os.path.join(args.data_dir, 'cifar100/'), args)
    N_SAMPLES_PER_CLASS = trainset.img_num_list
        
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last= args.loss_fn == 'ncl', pin_memory=True, sampler=None)
    testloader = data.DataLoader(testset, batch_size=args.batch_size*4, shuffle=True, num_workers=args.workers, pin_memory=True) 
    
    if args.cmo:
        cls_num_list = N_SAMPLES_PER_CLASS
        cls_weight = 1.0 / (np.array(cls_num_list))
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        labels = trainloader.dataset.targets
        samples_weight = np.array([cls_weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(labels), replacement=True)
        weighted_trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=sampler)
    else:
        weighted_trainloader = None
    

    # Model
    print ("==> creating {}".format(args.network))
    model = get_model(args, N_SAMPLES_PER_CLASS)
    train_criterion = get_loss(args, N_SAMPLES_PER_CLASS)
    criterion = nn.CrossEntropyLoss() # For test, validation 
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args,optimizer)

    teacher = load_model(args)


    train = get_train_fn(args)
    validate = get_valid_fn(args)
    update_score = get_update_score_fn(args)
    
    start_time = time.time()
    
    test_accs = []
    for epoch in range(args.epochs):
        
        lr = adjust_learning_rate(optimizer, epoch, scheduler, args)
        if args.cuda:
            if epoch % args.update_epoch == 0:
                curr_state, label = update_score(trainloader, model, N_SAMPLES_PER_CLASS, posthoc_la = args.posthoc_la, num_test = args.num_test, accept_rate = args.accept_rate)

            if args.verbose:
                if epoch == 0:
                    maps = np.zeros((args.epochs,args.num_class))
                maps = plot_score_epoch(curr_state,label, epoch, maps, args.out)
        train_loss = train(args, trainloader, model, optimizer,train_criterion, epoch, weighted_trainloader, teacher) 


        test_loss, test_acc, test_cls = validate(args, testloader, model, criterion, N_SAMPLES_PER_CLASS,  num_class=args.num_class, mode='test Valid')

        if best_acc <= test_acc:
            best_acc = test_acc
            many_best = test_cls[0]
            med_best = test_cls[1]
            few_best = test_cls[2]
            # Save models
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model['model'].state_dict() if args.loss_fn == 'ncl' else model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, epoch + 1, args.out)
        test_accs.append(test_acc)

        args.logger(f'Epoch: [{epoch+1} | {args.epochs}]', level=1)
        if args.cuda:
            args.logger(f'Max_state: {int(torch.max(curr_state))}, min_state: {int(torch.min(curr_state))}', level=2)
        args.logger(f'[Train]\tLoss:\t{train_loss:.4f}', level=2)
        args.logger(f'[Test ]\tLoss:\t{test_loss:.4f}\tAcc:\t{test_acc:.4f}', level=2)
        args.logger(f'[Stats]\tMany:\t{test_cls[0]:.4f}\tMedium:\t{test_cls[1]:.4f}\tFew:\t{test_cls[2]:.4f}', level=2)
        args.logger(f'[Best ]\tAcc:\t{np.max(test_accs):.4f}\tMany:\t{100*many_best:.4f}\tMedium:\t{100*med_best:.4f}\tFew:\t{100*few_best:.4f}', level=2)
        args.logger(f'[Param]\tLR:\t{lr:.8f}', level=2)
    
    end_time = time.time()
    
    # loading saved stage1
    load_checkpoint = True
    args.logger(f' Results after stage 1 from checkpoint {args.out}/checkpoint.pth.tar', level=1)
    if load_checkpoint:
        checkpoint = torch.load(f'{args.out}/checkpoint.pth.tar')  # Replace with the actual path to your checkpoint
        
        #
        if args.loss_fn == 'ncl':
            model['model'].load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        # Print the final results
        test_loss, test_acc, test_cls = validate(args, testloader, model, criterion, N_SAMPLES_PER_CLASS,  num_class=args.num_class, mode='test Valid')
  
        args.logger(f'Acc (test):\t{test_acc}', level=2)
        args.logger(f'[Stats]\tMany:\t{test_cls[0]:.4f}\tMedium:\t{test_cls[1]:.4f}\tFew:\t{test_cls[2]:.4f}', level=2)
    else:
        args.logger(f'Final performance...', level=1)
        args.logger(f'best bAcc (test):\t{np.max(test_accs)}', level=2)
        args.logger(f'best statistics:\tMany:\t{many_best}\tMed:\t{med_best}\tFew:\t{few_best}', level=2)
        args.logger(f'Training Time: {hms_string(end_time - start_time)}', level=1)
    model = Second_stage_classifier_alignment(model,trainloader)
    
    test_loss, test_acc, test_cls = validate(args, testloader, model, criterion, N_SAMPLES_PER_CLASS,  num_class=args.num_class, mode='test Valid')
    many_best = test_cls[0]
    med_best = test_cls[1]
    few_best = test_cls[2]
    args.logger(f'After second stage finetuning:', level=1)
    args.logger(f'Acc (test):\t{test_acc}', level=2)
    args.logger(f'[Stats]\tMany:\t{test_cls[0]:.4f}\tMedium:\t{test_cls[1]:.4f}\tFew:\t{test_cls[2]:.4f}', level=2)
       
    
    if args.verbose:
        args.logger.map_save(maps)
        
def Second_stage_classifier_alignment(model, train_loader):
    clipgrad=10000
    prototypes, Majority_class_features, num_of_classes, distrib = calculate_prototypes(model, train_loader)
    # breakpoint()
    # samples = add_feature_distance(prototypes, Majority_class_features)
    for parameter in model.parameters():
        parameter.requires_grad = False

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    for parameter in model.linear.parameters():
        parameter.requires_grad = True
            
    print("trainable_parameters_list....")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            
    params = model.linear.parameters()
    lr = 0.1
    momentum=0.9
    wd=2e-4
    optimizer_classifier_tune = torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
    
    # breakpoint()
    
    model.train()
    for e in range(500):
        
        samples = distrib.rsample().cuda().float()
        
        # samples = add_feature_distance(prototypes, Majority_class_features).cuda().float()
        targets = torch.arange(num_of_classes).cuda()
        
        outputs = model.forward_linear(samples)
        loss = nn.CrossEntropyLoss(None)(outputs, targets.long())
        optimizer_classifier_tune.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipgrad)
        optimizer_classifier_tune.step()
        
        if e % 50 == 0:
            args.logger(f"classifer tuning: epoch ---> {e}", level=1)
            args.logger(f'[Train]\tLoss:\t{loss:.4f}', level=2)
    
    return model
        
        

def add_feature_distance(prototypes, Majority_class_features):
    random_indices = torch.randperm(Majority_class_features.size(0))[:prototypes.shape[0]]

    aug_prototypes = prototypes + Majority_class_features[random_indices]
        # breakpoint()
    return aug_prototypes

def calculate_prototypes(model, train_loader):
    features_list = []
    labels = []
    norm_feature_bank = []
    model.eval()
    
    
    with torch.no_grad():
        for batch_idx, data_tuple in enumerate(train_loader):
            inputs_b = data_tuple[0]
            targets_b = data_tuple[1]
            indexs = data_tuple[2]
            inputs_b = inputs_b.cuda(non_blocking=True)
            targets_b = targets_b.cuda(non_blocking=True)
            _, features = model(inputs_b)
            labels.append(targets_b.cpu())
            features_list.append(features.cpu())
    
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels, dim=0)
    num_of_unique_classes = torch.unique(labels)
    num_of_classes = len(num_of_unique_classes)
    
    # breakpoint()
    
    prototype = []
    radius = []
    class_label = []
    cov_list = []
    num_of_samples = []
    for class_index in range(num_of_classes):
        

        data_index = (labels == class_index).nonzero()
        embedding = features[data_index.squeeze(-1)]
        embedding = F.normalize(embedding, p=2, dim=-1)
        feature_class_wise = embedding.numpy()
        cov = np.cov(feature_class_wise.T)
        # cov_torch = torch.cov(embedding.T)
        # radius.append(np.trace(cov)/64)
        
        print('class index', class_index, 'number of samples',data_index.shape[0])
        num_of_samples.append(data_index.shape[0])
        embedding_mean = embedding.mean(0)
        prototype.append(embedding_mean)
        cov_list.append(torch.tensor(cov))
        # cov_list.append(cov_torch)
        
    proto_list = torch.stack(prototype, dim=0)
    num_of_samples = torch.tensor(num_of_samples)
    class_id_most_samples = torch.argmax(num_of_samples)
    
    cov_cls_ms_major = cov_list[class_id_most_samples]
    cov_cls_ms = cov_cls_ms_major.repeat(num_of_classes, 1, 1)
    mean = proto_list
    distrib = MultivariateNormal(loc=mean.double(), covariance_matrix=cov_cls_ms.double())
 
    # breakpoint()
    
    data_index = (labels == class_id_most_samples).nonzero()
    embedding = features[data_index.squeeze(-1)]
    embedding = F.normalize(embedding, p=2, dim=-1)
    norm_feature_bank = embedding - embedding.mean(0)
    # breakpoint()
    return proto_list, norm_feature_bank, num_of_classes, distrib

        
    
    

if __name__ == '__main__':
    main()