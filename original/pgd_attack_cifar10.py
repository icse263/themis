from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import pandas as pd
#python pgd_attack_cifar10.py --diff --source-model-path ./checkpoints/model_cifar_wrn.pt --target-model-path ./checkpoints/model-wideres-epoch20.pt

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white-box-attack', action='store_true', default=False,
                    help='whether perform white-box attack')
parser.add_argument('--gen', action='store_true', default=False,
                    help='generate dataset')
parser.add_argument('--diff', action='store_true', default=False,
                    help='computing diff in activation')
parser.add_argument('--heat', action='store_true', default=False,
                    help='plot heat map')
parser.add_argument('--var', action='store_true', default=False,
                    help='plot variance graph')
parser.add_argument('--bar', action='store_true', default=False,
                    help='plot bar chart')

args = parser.parse_args()

# settings
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
advset = torch.load("../data/cifar10_adv_black.pt")
adv_loader = torch.utils.data.DataLoader(advset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)


    err_pgd = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err pgd black-box: ', err_pgd)
    return err, err_pgd

# def _pgd_blackbox(model_target,
#                   model_source,
#                   X,
#                   y,
#                   epsilon=args.epsilon,
#                   num_steps=args.num_steps,
#                   step_size=args.step_size):
#     out = model_target(X)
#     print("X: " + str(X.data[0][0][0]))
#     err = (out.data.max(1)[1] != y.data).float().sum()
#     X_pgd = Variable(X.data, requires_grad=True)
#     if args.random:
#         random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
#         X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
#
#     for _ in range(int(num_steps)):
#         opt = optim.SGD([X_pgd], lr=1e-3)
#         opt.zero_grad()
#         with torch.enable_grad():
#             loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
#         loss.backward()
#         eta = step_size * X_pgd.grad.data.sign()
#         X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
#         eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
#         X_pgd = Variable(X.data + eta, requires_grad=True)
#         print("X_pgd before clamp: " + str(X_pgd.data[0][0][0]))
#         X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
#         print("X_pgd after clamp: " + str(X_pgd.data[0][0][0]))
# #     return X_pgd

def _pgd_blackbox_gen(model_target,
                  model_source,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model_target(X)
    print("X: " + str(X.data[0][0][0]))
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(int(num_steps)):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        print("X_pgd before clamp: " + str(X_pgd.data[0][0][0]))
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        print("X_pgd after clamp: " + str(X_pgd.data[0][0][0]))
    return X_pgd

def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0
    if args.gen:
        all_datasets = []
        print("Start generating data")
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            X_pgd = _pgd_blackbox_gen(model_target, model_source, X, y)
            all_datasets.append(torch.utils.data.TensorDataset(X_pgd, y))
            # break
        final_dataset = torch.utils.data.ConcatDataset(all_datasets)
        print("final_dataset: " + str(final_dataset))
        torch.save(final_dataset, "../data/cifar10_adv_black.pt")
    elif args.diff:
        print("start computing differences")
        dateTimeObj = datetime.now()
        date_name = str(dateTimeObj).replace(' ', '-').replace(':', '-').replace('.', '-')
        # print("advset[0]: " + str(advset[0]))
        # return
        heat_dict = {}
        row_max = 2000
        row_count = 0
        act_list_len = 0
        print("len(advset): " + str(len(advset)))
        for i in range(len(advset)):
            d_adv, t_adv = advset[i][0].to(device), advset[i][1].to(device)
            # pgd attack
            X_adv, y_adv = Variable(d_adv), Variable(t_adv)
            out_adv, act_list_adv = model_target.forward_act(X_adv.unsqueeze(0))
            corr_adv = out_adv.data.max(1)[1] == y_adv.data
            # print("y test: " + str(target_test))
            if not corr_adv:
                d_test, t_test = testset[i][0].to(device), torch.tensor(testset[i][1]).to(device)
                    # pgd attack
                X_test, y_test = Variable(d_test), Variable(t_test)
                out_test, act_list_test = model_target.forward_act(X_test.unsqueeze(0))
                corr_test = out_test.data.max(1)[1] == y_test.data
                if corr_test:
                    print("now in row: " + str(row_count))
                    act_list_len = len(act_list_test)
                    for j in range(act_list_len):
                        sub = torch.absolute(torch.subtract(act_list_test[j],act_list_adv[j]))
                        flatten_np = sub.cpu().data.numpy().flatten()
                        if not j in heat_dict:
                            heat_dict[j] = [flatten_np]
                        else:
                            heat_dict[j].append(flatten_np)
                    row_count = row_count + 1
            if row_count == row_max:
                if args.heat:
                    for j in range(act_list_len):
                       fig, ax = plt.subplots()
                       # plt.imshow(np.reshape(flatten_np, (-1, len(flatten_np))), cmap='hot', interpolation='nearest')
                       ax = sns.heatmap(heat_dict[j])
                       plt.savefig("fig/cifar_10_" + str(j) + "," + str(row_max) + "," + date_name + ".pdf")
                if args.var:
                    for j in range(1):
                    # for j in range(act_list_len):
                        print("plotting "+ str(j))
                        df = pd.DataFrame(heat_dict[j])
                        if args.bar:
                            df_var = df.var().to_frame()
                            fig, ax = plt.subplots()
                            ax = df_var.hist(bins=12)
                            plt.savefig("fig/bin_cifar_10_" + str(j) + "," + str(row_max) + "," + date_name + ".pdf")
                        else:
                            print("combing df and comparing ranks")
                            all_df_rank = []
                            all_df_var = []
                            h_rows = [100,500,1000,1500,2000]
                            for h_row in h_rows:
                                df_var = df.head(h_row).var().to_frame()
                                df_var.columns = ['a_' + str(h_row)]
                                df_rank = df_var.rank()
                                df_rank.columns = ['a_' + str(h_row)]
                                # df_var.sort_values(by=['a_' + str(h_row)], ascending=False)
                                all_df_var.append(df_var)
                                all_df_rank.append(df_rank)

                            show_df_rank = pd.concat(all_df_rank,axis=1)
                            show_df_var = pd.concat(all_df_var, axis=1)

                            show_df_rank_sort = show_df_rank.sort_values(by=['a_' + str(h_rows[0])], ascending=False)
                            show_df_var_sort = show_df_var.sort_values(by=['a_' + str(h_rows[0])], ascending=False)

                            show_df_rank_sort.to_csv("fig/adv_rank_" + str(row_max) + "," + str(date_name) + ".csv", index = True, header=True)
                            show_df_var_sort.to_csv("fig/adv_var_" + str(row_max) + "," + str(date_name) + ".csv",index=True, header=True)
                            print(show_df_rank_sort)
                            print(show_df_var_sort)
                                # fig, ax = plt.subplots()
                                # ax = df_var.hist(bins=12)
                        # df_var.columns = ['a']
                        # print(df_var.sort_values(by=['a'],ascending=False))
                        # plt.savefig("fig/cifar_10_" + str(j) + "," + str(row_max) + "," + date_name + ".pdf")
            if row_count == row_max:
                break
            # break


    else:
        for data, target in test_loader:
            print("start the original attack")
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y)
            robust_err_total += err_robust
            natural_err_total += err_natural
        print('natural_err_total: ', natural_err_total)
        print('robust_err_total: ', robust_err_total)

def main():

    if args.white_box_attack:
        # white-box attack
        print('pgd white-box attack')
        model = WideResNet().to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        eval_adv_test_whitebox(model, device, test_loader)
    else:
        # black-box attack
        print('pgd black-box attack')
        model_target = WideResNet().to(device)
        model_target.load_state_dict(torch.load(args.target_model_path, map_location=device))
        model_source = WideResNet().to(device)
        model_source.load_state_dict(torch.load(args.source_model_path, map_location=device))
        eval_adv_test_blackbox(model_target, model_source, device, test_loader)


if __name__ == '__main__':
    main()