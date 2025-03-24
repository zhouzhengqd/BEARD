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
from eva_utils import parser_bool, downscale, epoch_no_loader, get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, number_sign_augment, padding_augment,augment,compute_std_mean,epoch_blackbox
import torchnet
import torch.nn.functional as F
import pickle
import torchattacks
from tqdm import tqdm
import time
import logging


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def setup_logging(log_dir, model, dataset, ipc):
   
    os.makedirs(log_dir, exist_ok=True)
    
  
    log_file = os.path.join(log_dir, f"{model}_{dataset}_{ipc}_evaluation_log.txt")

    logging.basicConfig(
        filename=log_file,            
        level=logging.INFO,            
        format='%(asctime)s - %(levelname)s - %(message)s'  
    )
    logging.info("Logging setup complete.")

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")

def evaluate_synset(it_eval, net, eval_nets, testloader, args, partial=None, aug=False, target_attack=True, test_attack=None):
    # eval_accuracies = {name: 0 for name in eval_nets.keys()}
    net = net.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    loss_test, acc_test, eval_accuracies= epoch_blackbox('test', testloader, net, eval_nets,optimizer, criterion, args, aug = aug, partial=partial, attack=test_attack, target_attack=target_attack)

    # for name, model in eval_nets.items():
    #     model.eval()
    #     with torch.no_grad():
    #         eval_output = model(img)
    #         eval_acc = np.sum(np.equal(np.argmax(eval_output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))
    #         eval_accuracies[name] += eval_acc

    #     # 归一化准确率
    # eval_accuracies = {name: acc / lab.shape[0] for name, acc in eval_accuracies.items()}


    print("Test Attack Type:"+test_attack)
    print("Target Attack:"+str(target_attack))
    print('%s Evaluate_%02d: epoch = %04d, test acc = %.4f' % (get_time(), it_eval, Epoch, acc_test))


    return net, acc_test,eval_accuracies



def main_train():
    global best_acc

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--method', type=str, default='DM', help='DC/DSA/DM')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='./data/', help='dataset path')
    parser.add_argument('--save_path', type=str, default='./model_pool/', help='path to save results')
    parser.add_argument('--eval_interval', type=int, default=100, help='outer loop for network update')
    parser_bool(parser, 'syn_ce', True)
    parser.add_argument('--ce_weight', type=float, default=0.1, help='outer loop for network update')
    parser_bool(parser, 'aug', True)
    parser.add_argument('--train_attack', default='None', action='store_true', help='Enable Train PGD')
    parser.add_argument('--test_attack', default='None', action='store_true', help='Enable Test PGD')
    parser.add_argument('--load_file', type=str,default='./model_pool/BACON/AT/BACON_ConvNet_CIFAR10_1_None_0.3139.pth', help='Path to the data file')
    parser.add_argument('--load_file_model_DC', type=str,default='./model_pool/DC/DC_ConvNet_CIFAR10_1_None.pth', help='Path to the data file')
    parser.add_argument('--load_file_model_DSA', type=str,default='./model_pool/DSA/DSA_ConvNet_CIFAR10_1_None_0.2823.pth', help='Path to the data file')
    parser.add_argument('--load_file_model_MTT', type=str,default='./model_pool/MTT/MTT_ConvNet_CIFAR10_1_None_0.4344.pth', help='Path to the data file')
    parser.add_argument('--load_file_model_DM', type=str,default='./model_pool/DM/DM_ConvNet_CIFAR10_1_None.pth', help='Path to the data file')
    parser.add_argument('--load_file_model_IDM', type=str,default='./model_pool/IDM/single_gpu/IDM_ConvNet_CIFAR10_50_None_0.6691.pth', help='Path to the data file')
    parser.add_argument('--load_file_model_BACON', type=str,default='./model_pool/BACON/BACON_ConvNet_CIFAR10_1_None_0.4495.pth', help='Path to the data file')
    parser.add_argument('--load_file_model_ROME', type=str,default='./model_pool/ROME/ROME_ConvNet_CIFAR10_1_None.pth', help='Path to the data file')
    parser_bool(parser, 'src_dataset', False)
    parser_bool(parser, 'target_attack', default=[True,False])
    parser_bool(parser, 'pgd_eva', False)
    parser.add_argument('--num_eval', type=int, default=0, help='the number of evaluating randomly initialized models')
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

    # if not os.path.exists(args.save_path):
    #     os.mkdir(args.save_path)

    log_dir = args.save_path
    if args.pgd_eva ==True:
        setup_logging(log_dir, args.pgd_eva, args.dataset, args.ipc)
    else:
        setup_logging(log_dir, args.model, args.dataset, args.ipc)

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_interval).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, trainloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, 'ConvNet', args.model)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []
    it = 1
    for model_eval in model_eval_pool:
        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, it))

        print('DSA augmentation strategy: \n', args.dsa_strategy)
        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

        infos = {attack: {'true': {'acc': [], 'time': []}, 'false': {'acc': [], 'time': []}} for attack in args.test_attack}

        # print(accs)
        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
        load_model(net_eval, args.load_file)

        eval_nets = {
            "DC": get_network(model_eval, channel, num_classes, im_size).to(args.device),
            "DSA": get_network(model_eval, channel, num_classes, im_size).to(args.device),
            "MTT": get_network(model_eval, channel, num_classes, im_size).to(args.device),
            "DM": get_network(model_eval, channel, num_classes, im_size).to(args.device),
            "IDM": get_network(model_eval, channel, num_classes, im_size).to(args.device),
            "BACON": get_network(model_eval, channel, num_classes, im_size).to(args.device),
            "ROME": get_network(model_eval, channel, num_classes, im_size).to(args.device),
        }

        # 加载模型参数
        load_model(eval_nets["DC"], args.load_file_model_DC)
        load_model(eval_nets["DSA"], args.load_file_model_DSA)
        load_model(eval_nets["MTT"], args.load_file_model_MTT)
        load_model(eval_nets["DM"], args.load_file_model_DM)
        load_model(eval_nets["IDM"], args.load_file_model_IDM)
        load_model(eval_nets["BACON"], args.load_file_model_BACON)
        load_model(eval_nets["ROME"], args.load_file_model_ROME)

        for attack in args.test_attack:
            for target_attack in args.target_attack:
                if args.num_eval == 0:
                    start_time = time.time()
                    _, acc_test,eval_accuracies = evaluate_synset(it, net_eval, eval_nets, testloader, args, aug=args.aug, target_attack = target_attack, test_attack=attack)
                    end_time = time.time()
                    attack_time = end_time - start_time
                    infos[attack][str(target_attack).lower()]['acc'].append(acc_test)
                    infos[attack][str(target_attack).lower()]['time'].append(attack_time)
                    infos[attack][str(target_attack).lower()].setdefault('eval_accuracies', []).append(eval_accuracies)

                else:
                    for i in range(args.num_eval):
                        print("current iteration: ", i)
                        _, acc_test,eval_accuracies = evaluate_synset(it, net_eval, eval_nets, testloader, args, aug=args.aug, target_attack = target_attack, test_attack=attack)
                        infos[attack][str(target_attack).lower()].append(acc_test)
                        infos[attack][str(target_attack).lower()].setdefault('eval_accuracies', []).append(eval_accuracies)

        for attack in args.test_attack:
            acc_target = infos[attack]['true']['acc']
            acc_non_target = infos[attack]['false']['acc']
            time_target = float(infos[attack]['true']['time'][0])
            time_non_target = float(infos[attack]['false']['time'][0])
            mean_target, std_target = compute_std_mean(acc_target)
            mean_non_target, std_non_target = compute_std_mean(acc_non_target)
            eval_acc_info_target = infos[attack]['true']['eval_accuracies']
            eval_acc_info_non_target  = infos[attack]['false']['eval_accuracies']
            print("%s attack on %s: final acc is:  target_attack: %.2f +- %.2f, attack time is %.2f; non_target_attack: %.2f +- %.2f, attack time is %.2f, dataset: %s, IPC: %s, DSA:%r, num_eval: %d, aug:%s , model: %s, attack_type: %s, target_attack: %s, src_dataset: %s, eval_accuracies_target: {%s}, eval_accuracies_non_target: {%s}"%( 
                        attack,
                        args.method,
                        mean_target * 100, std_target * 100,
                        time_target,
                        mean_non_target * 100, std_non_target * 100,
                        time_non_target,
                        args.dataset, 
                        args.ipc,
                        args.dsa,
                        args.num_eval,
                        args.aug,
                        args.model,
                        args.test_attack,
                        args.target_attack,
                        args.src_dataset,
                        eval_acc_info_target,
                        eval_acc_info_non_target
                    ))
            log_message = ("%s attack on %s: final acc is:  target_attack: %.2f +- %.2f, attack time is %.2f; non_target_attack: %.2f +- %.2f, attack time is %.2f, dataset: %s, IPC: %s, DSA:%r, num_eval: %d, aug:%s , model: %s, attack_type: %s, target_attack: %s, src_dataset: %s, eval_accuracies_target: {%s}, eval_accuracies_non_target: {%s}"%( 
                        attack,
                        args.method,
                        mean_target * 100, std_target * 100,
                        time_target,
                        mean_non_target * 100, std_non_target * 100,
                        time_non_target,
                        args.dataset, 
                        args.ipc,
                        args.dsa,
                        args.num_eval,
                        args.aug,
                        args.model,
                        args.test_attack,
                        args.target_attack,
                        args.src_dataset,
                        eval_acc_info_target,
                        eval_acc_info_non_target
                    ))
    
            logging.info(log_message)




if __name__ == "__main__":
    main_train()