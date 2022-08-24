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

#generate adversarial exampels:
#python gen_adv_cifar10.py --white-box-attack False

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
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
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')

args = parser.parse_args()

# settings
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


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

    for _ in range(int(num_steps)):
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



def _pgd_blackbox(model_target,
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
    # np.save("data_attack/cifar10_X_adv.npy", X_pgd.detach().cpu().numpy())


def eval_adv_test_whitebox(model, device, test_loader):
    model.eval()
    all_datasets = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        X_pgd = _pgd_whitebox(model, X, y)
        all_datasets.append(torch.utils.data.TensorDataset(X_pgd, y))
        break
    final_dataset = torch.utils.data.ConcatDataset(all_datasets)
    print("final_dataset: " + str(final_dataset))
    torch.save(final_dataset, "../data/cifar10_adv_white.pt")

def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    model_target.eval()
    model_source.eval()
    all_datasets = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        X_pgd = _pgd_blackbox(model_target, model_source, X, y)
        all_datasets.append(torch.utils.data.TensorDataset(X_pgd, y))
        # break
    final_dataset = torch.utils.data.ConcatDataset(all_datasets)
    print("final_dataset: " + str(final_dataset))
    torch.save(final_dataset, "../data/cifar10_adv_black.pt")


def main():
    print("args.white_box_attack: " + str(args.white_box_attack))
    if args.white_box_attack == True:
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
