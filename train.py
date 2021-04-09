import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler_torch
from torch.utils.data import DataLoader

import preprocess as prep
import networks as network
import lr_schedule
import loss

from torchvision import datasets
optim_dict = {'SGD':optim.SGD, 'Adam':optim.Adam}

def f1Loss(predict, all_label):
    predict_list = predict.numpy()
    all_label_list = all_label.numpy()
    TP = 0
    FP = 0
    FN = 0
    for number in range(len(all_label_list)):
        if predict_list[number] == 1:
            if all_label_list[number] == 1:
                TP = TP + 1
            else:
                FP = FP + 1
        elif all_label_list[number] == 1:
            FN = FN + 1

    P = float(TP) / (TP + FP) if (TP + FP != 0) else 0
    R = float(TP) / (TP + FN) if (TP + FN != 0) else 0
    F = float((2 * P * R) / (P + R)) if P + R != 0 else 0
    return F


def classification(config):

    ## load dataset
    prep_config   = config['prep']
    transforms_s  = prep.normalize(prep_config['source'])
    transforms_t  = prep.normalize(prep_config['target'])

    data_config = config['data']
    source_config = data_config['source']
    target_config = data_config['target']

    train_dataset = datasets.ImageFolder(source_config['path'], transform=transforms_s)
    test_dataset  = datasets.ImageFolder(target_config['path'], transform=transforms_t)

    train_loader = DataLoader(train_dataset,
                                batch_size=source_config['batch_size'],
                                shuffle=True,
                                num_workers=2,
                                drop_last=True)
    test_loader = DataLoader(test_dataset,
                                batch_size=target_config['batch_size'],
                                shuffle=True,
                                num_workers=2,
                                drop_last=True)
    class_num = 2

    ## set loss
    class_criterion     = nn.CrossEntropyLoss()

    loss_config         = config['loss']
    transfer_criterion   = loss.loss_dict[loss_config['name']]

    ## set base network
    net_config      = config['network']
    base_network    = network.network_dict[net_config['name']]()
    if net_config['bottleneck']:
        bottleneck_layer = nn.Linear(base_network.output_num(), net_config['bottleneck_dim'])
        classifier_layer = nn.Linear(bottleneck_layer.out_features, class_num)
    else:
        classifier_layer = nn.Linear(base_network.output_num(), class_num)

    for param in base_network.parameters():
        param.requires_grad = True

    ## initialization
    if net_config['bottleneck']:
        bottleneck_layer.weight.data.normal_(0, 0.005)
        bottleneck_layer.bias.data.fill_(0.1)
        bottleneck_layer = nn.Sequential(bottleneck_layer, nn.ReLU(), nn.Dropout(0.5))
    classifier_layer.weight.data.normal_(0, 0.01)
    classifier_layer.bias.data.fill_(0.0)

    ## set gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device : {device}')
        # multi gpu
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        classifier_layer = nn.DataParallel(classifier_layer)
        base_network = nn.DataParallel(base_network)
        if net_config['bottleneck']:
            bottleneck_layer = nn.DataParallel(bottleneck_layer)
    classifier_layer.to(device)
    base_network.to(device)
    bottleneck_layer.to(device)
    
    # set optimizer
    if net_config['bottleneck']:
        parameter_list = [
            {'params':base_network.parameters(), 'lr':10},
            {'params':bottleneck_layer.parameters(), 'lr':10},
            {'params':classifier_layer.parameters(), 'lr':10}]
    else:
        parameter_list = [
            {'params':base_network.parameters(), 'lr':10},
            {'params':classifier_layer.parameters(), 'lr':10}
        ]
    optimizer_config = config['optimizer']
    optimizer        = optim_dict[optimizer_config['name']](parameter_list, **(optimizer_config['optim_params']))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    schedule_param = optimizer_config['lr_param']
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config['lr_type']]
    

    ## train
    # train_classifier_losses, train_total_losses, test_losses = [], [], []
    running_loss = 0
    total_loss   = 0    
    steps        = 0
    print_every  = 5
    test_accuracies = []
    test_f1 = []
    epochs = config['epoch']
    for epoch in range(epochs):   
        iter_target = iter(test_loader)
        for inputs_source, labels_source in train_loader:
            inputs_source  = inputs_source.to(device)
            labels_source  = labels_source.to(device)

            if steps % len(test_loader) == 0:
                iter_target = iter(test_loader)
            
            inputs_target, _labels_target = iter_target.next()
            inputs_target    = inputs_target.to(device)

            if net_config["bottleneck"]:
                bottleneck_layer.train(True)
            base_network.train(True)
            classifier_layer.train(True)

            optimizer = lr_scheduler(param_lr, optimizer, steps, **schedule_param)
            optimizer.zero_grad()
            steps += 1

            inputs = torch.cat((inputs_source, inputs_target), dim=0)
            features = base_network(inputs)
            if net_config["bottleneck"]:
                features = bottleneck_layer(features)

            # classifier loss
            outputs = classifier_layer(features)
            classifier_loss = class_criterion(outputs.narrow(0, 0, int(inputs.size(0)/2)), labels_source)
            running_loss += classifier_loss.item()

            # transfer loss
            if loss_config["name"] == "DAN" or loss_config["name"] == "DAN_Linear":
                transfer_loss = transfer_criterion(features.narrow(0, 0, int(features.size(0)/2)), features.narrow(0, int(features.size(0)/2), int(features.size(0)/2)))
            
            # total loss
            total_loss = classifier_loss + loss_config["tradeoff"] * transfer_loss
            total_loss += total_loss
            
            total_loss.backward()
            optimizer.step()

            if steps % print_every == 0:
                test_loss = 0
                accuracy  = 0

                base_network.train(False)
                classifier_layer.train(False)
                if net_config['bottleneck']:
                    bottleneck_layer.train(False)

                with torch.no_grad():
                    all_out   = torch.empty(0)
                    all_label = torch.empty(0)
                    for inputs_targets, labels_targets in test_loader:
                        inputs_targets = inputs_targets.to(device)
                        labels_targets = labels_targets.to(device)

                        features = base_network(inputs_targets)
                        if net_config["bottleneck"]:
                            features = bottleneck_layer(features)
                        features   = classifier_layer(features)
                        batch_loss = class_criterion(features, labels_targets)
                        
                        test_loss += batch_loss.item()

                        top_p, top_class = features.topk(1, dim=1)
                        
                        equals = top_class == labels_targets.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                        all_out   = torch.cat((all_out, top_class.cpu()), 0)
                        all_label = torch.cat((all_label, labels_targets.cpu()), 0)

                test_accuracies.append(accuracy/len(test_loader))
                f1 = f1Loss(all_out, all_label)
                test_f1.append(f1)

                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Class loss: {running_loss/print_every:.3f}.. "
                    f"Train Total loss : {total_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(test_loader):.3f}.. "
                    f"Test accuracy: {accuracy/len(test_loader):.3f}.. "
                    f"Test f1 score: {f1:.3f}..")
                running_loss = 0
                total_loss   = 0

    mean_accuracy = 0
    mean_f1       = 0
    for accuracy, f1 in zip(test_accuracies, test_f1):
        mean_accuracy += accuracy
        mean_f1       += f1

    print(f"mean accuracy = {mean_accuracy/len(test_accuracies)}")
    print(f"mean f1 score = {mean_f1/len(test_f1)}")

def arg_parse():
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--source', type=str, nargs='?', default='ant-1.6', help="source data")
    parser.add_argument('--target', type=str, nargs='?', default='poi-3.0', help="target data")
    parser.add_argument('--network', type=str, nargs='?', default='ResNet50Fc', help="network name")
    parser.add_argument('--loss_name', type=str, nargs='?', default='DAN_Linear', help="loss name")
    parser.add_argument('--tradeoff', type=float, nargs='?', default=1, help="tradeoff")
    parser.add_argument('--bottleneck', type=int, nargs='?', default=1, help="whether to use bottleneck")
    return parser.parse_args()
    

if __name__ =="__main__":

    args = arg_parse()
    path = 'data/img/gray_img/'

    config = {}
    config['epoch'] = 50
    config['prep'] = {
        'source':{'name':'source', 'resize_size':256, 'crop_size':224},
        'target':{'name':'target', 'resize_size':256, 'crop_size':224}
    }
    
    config['data'] = {
        'source':{
            'path':path + args.source,
            'batch_size': 64
            },
        'target':{
            'path':path + args.target,
            'batch_size': 64
            }
    }

    config['loss'] = {'name':args.loss_name, 'tradeoff':args.tradeoff}
    config['network'] = {
        'name':args.network,
        'bottleneck':args.bottleneck,
        'bottleneck_dim':256
    }

    config['optimizer'] = {
        'name':'SGD', 'optim_params':{'lr':0.05, 'momentum':0.9, 'weight_decay':0.0005, 'nesterov':True},
        'lr_type':'inv', 'lr_param':{'init_lr':0.0003, 'gamma':0.0003, 'power':0.75}
    }

    # print(config['loss'])

    print(config)
    classification(config)

