import os
import json
import argparse
import random
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from eva_utils import parser_bool, downscale, epoch_no_loader, get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, number_sign_augment, padding_augment
import torchnet
import torch.nn.functional as F
import pickle
import torchattacks

best_acc = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# torch.cuda.device_count()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    global best_acc

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet_GBN', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--method', type=str, default='DM', help='DC/DSA/DM')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='./data/', help='dataset path')
    parser.add_argument('--save_path', type=str, default='./evaluation/', help='path to save results')
    parser.add_argument('--eval_interval', type=int, default=100, help='outer loop for network update')
    parser_bool(parser, 'syn_ce', False)
    parser.add_argument('--ce_weight', type=float, default=0.1, help='outer loop for network update')
    parser_bool(parser, 'aug', False)
    parser.add_argument('--data_file', type=str,default='./dataset_pool/BACON_cifar10_ipc50_conv.pt', help='Path to the data file')
    parser.add_argument('--train_attack', default='None', action='store_true', help='Enable Train PGD')
    parser.add_argument('--test_attack', default='None', action='store_true', help='Enable Test PGD')
    parser_bool(parser, 'src_dataset', False)
    parser_bool(parser, 'target_attack', True)
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    print('Configuration:')
    print(json.dumps(vars(args), indent=4))

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_interval).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, trainloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, 'ConvNet', args.model)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []
    data_file = args.data_file

    if args.method == "MTT":
        image_syn = torch.load(data_file+'images_best.pt')
        label_syn = torch.load(data_file+'labels_best.pt')
        # Load the .pt file
    else:
        checkpoint = torch.load(data_file)
        data_save = checkpoint['data']

        # Extract image_syn and label_syn
        image_syn, label_syn = data_save[0]

    # Ensure the tensors are on the correct device (e.g., GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_syn = image_syn.to(device)
    label_syn = label_syn.to(device)
    it = 1
    for model_eval in model_eval_pool:
        if args.dsa:
            args.dc_aug_param = None
            print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, it))

            print('DSA augmentation strategy: \n', args.dsa_strategy)
            print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
        else:
            args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc)
            print('DC augmentation parameters: \n', args.dc_aug_param)

        accs = []
        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
        if args.aug:
            image_syn_eval, label_syn_eval = number_sign_augment(image_syn_eval, label_syn_eval)
        _, acc_train, acc_test = evaluate_synset(it, net_eval, image_syn_eval, label_syn_eval, trainloader, testloader, args, tqdm_bar=True, train_attack=args.train_attack, test_attack=args.test_attack, src_dataset=args.src_dataset, target_attack=args.target_attack)
        accs.append(acc_test)
        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs), model_eval, np.mean(accs), np.std(accs)))

if __name__ == '__main__':
    main()
