from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import yaml
from models.wideresnet import *
import math
from models.resnet import *
from models.net_mnist import *
from models.small_cnn import *
from trades import trades_loss

#python train_trades.py --dataset mnist --save-freq 5
#trained model location: model-cifar-wideResNet
class retrain_o(object):
    def __init__(self, model_name, model,dataset,device,train_loader, attack, num_data, epsilon, log_interval=100,save_freq=10):
        with open(r'yaml/para.yaml') as file:
            para = yaml.load(file)
        self.model = model
        self.model_name = model_name
        self.dataset = dataset
        self.device = device
        self.learning_rate = para[model_name]["learning-rate"]
        self.step_size = para[model_name]["step-size"]
        # self.epsilon = para[model_name]["epsilon"]
        self.epsilon = epsilon
        self.beta = para[model_name]["beta"]
        self.momentum = para[model_name]["momentum"]
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        # self.epochs = para[model_name]["epochs"]
        self.train_loader = train_loader
        self.log_interval = log_interval
        self.save_freq = save_freq
        self.attack = attack
        self.num_data = num_data
        self.already_train = 0
        self.epochs = math.ceil(self.num_data / 1000)


    def start_train(self):
        print("epoch: " + str(self.epochs))#debug
        for epoch in range(1, self.epochs + 1):
            # adjust learning rate for SGD
            self.adjust_learning_rate(self.optimizer, epoch)

            self.train()

            # evaluation on natural examples
            print('================================================================')
            self.eval_train(self.model, self.device, self.train_loader)
            # self.eval_test(self.model, self.device, self.test_loader)
            print('================================================================')

            # save checkpoint

        torch.save(self.model.state_dict(),
                   os.path.join("checkpoints/",
                                "model_" + self.model_name +"_" + self.dataset + "_" + self.attack + "_" + str(self.epsilon) + "_" + str(self.num_data) + "-epoch{}.pt".format(epoch)))
        torch.save(self.optimizer.state_dict(),
                   os.path.join("checkpoints/",
                                self.model_name +"_" + self.dataset + "_" + self.attack + "_" + str(self.epsilon) + "_" + str(self.num_data) + '_checkpoint_epoch{}.tar'.format(epoch)))



    def train(self):
        self.model.train()
        print("loss function: original loss")
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data= torch.tensor(data, dtype=torch.float32)
            target=torch.tensor(target, dtype=torch.long) 
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            # calculate robust loss
            logits = self.model(data)
            #target = target.squeeze(1) #if dataset is drebin,there is a bug will be fixed in the future 
            loss_natural = F.cross_entropy(logits, target)
            loss = loss_natural
            loss.backward()
            self.optimizer.step()

            # print progress
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epochs, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))

    def eval_train(self,model, device, train_loader):
        model.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in train_loader:
                data= torch.tensor(data, dtype=torch.float32)
                target=torch.tensor(target, dtype=torch.long) 
                data, target = data.to(device), target.to(device)
                #target = target.squeeze(1) #if dataset is drebin,there is a bug will be fixed in the future 
                output = model(data)
                train_loss += F.cross_entropy(output, target, size_average=False).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss /= len(train_loader.dataset)
        print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
        training_accuracy = correct / len(train_loader.dataset)
        return train_loss, training_accuracy


    def eval_test(self,model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, size_average=False).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        test_accuracy = correct / len(test_loader.dataset)
        return test_loss, test_accuracy


    def adjust_learning_rate(self,optimizer, epoch):
        """decrease the learning rate"""
        lr = self.learning_rate
        if epoch >= 75:
            lr = self.learning_rate * 0.1
        if epoch >= 90:
            lr = self.learning_rate * 0.01
        if epoch >= 100:
            lr = self.learning_rate * 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    # parser.add_argument('--model-dir', required=True,
    #                     help='directory of model for saving checkpoint')
    parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                        help='save frequency')
    parser.add_argument('--adv', action='store_true', default=False,
                        help='Enable adversarial (Trades) training')
    parser.add_argument('--dataset', required=True,
                        help='what dataset')
    parser.add_argument('--mode', required=True,
                        help='train or retrain')

    args = parser.parse_args()
    with open(r'yaml/para.yaml') as file:
        para = yaml.load(file)
    # settings
    model_dir = "model_" + args.dataset
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if args.dataset == "cifar":
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=para[args.dataset]["batch-size"], shuffle=True,
                                                   **kwargs)
        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=para[args.dataset]["test-batch-size"],
                                                  shuffle=False, **kwargs)
    elif args.dataset == "mnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=para[args.dataset]["batch-size"], shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False,
                           transform=transforms.ToTensor()),
            batch_size=para[args.dataset]["test-batch-size"], shuffle=False, **kwargs)
    # init model, ResNet18() can be also used here for training
    if args.dataset == "cifar":
        model = WideResNet().to(device)
    elif args.dataset == "mnist":
        model = SmallCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=para[args.dataset]["lr"], momentum=para[args.dataset]["momentum"], weight_decay=para[args.dataset]["weight-decay"])

    for epoch in range(1, para[args.dataset]["epochs"] + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        train(args, model, device, train_loader, optimizer, epoch, args.adv)

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        is_adv = "clean"
        if args.adv:
            is_adv = "adv"

        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-' + args.dataset + "-" + is_adv + '-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, args.dataset + "-" + is_adv + '-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()
