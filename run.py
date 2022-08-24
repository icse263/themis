from __future__ import print_function
import argparse
import torch
from torchvision import datasets, transforms
# from keras.models import model_from_json, load_model, save_model
from models.wideresnet import *
from models.resnet import *
from models.small_cnn import *
from models.net_mnist import *
from models.densenet import *
from models.PDF import *
from models.Drebin import *
from PDF_dataset import DatasetFromCSV
from Drebin_dataset import DatasetFromDrebin
from models.model import *
import matplotlib
matplotlib.use('pdf')
import yaml
import input_add_noise as noise
import pgd_attack as pgd_a
import train_trades as tt

#--row_max 20 for MNIST because at most 20 adversarial misclassification for MNIST
#--row_max 100 for CIFAR because for resonable size fig
#CLEAN is without --adv
#Generate PGD data on TRADES: python run.py --dataset XXX --gen --adv
#Generate PGD data on CLEAN: python run.py --dataset XXX --gen
#df on TRADES: python run.py --dataset XXX --diff --var --row_max XXX --adv
#heat on TRADES: python run.py --dataset [cifar10,mnist] --diff --heat --row_max [100,20] --adv --attack [rand,pgd]

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=20, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003, type=float,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--white-box-attack', action='store_true', default=False,
                    help='whether perform white-box attack')
parser.add_argument('--eval', action='store_true', default=False,
                    help='to evaluate model adv acc')
parser.add_argument('--gen', action='store_true', default=False,
                    help='generate dataset')
parser.add_argument('--diff', action='store_true', default=False,
                    help='computing diff in activation')
parser.add_argument('--retrain', action='store_true', default=False,
                    help='see if retraining')
parser.add_argument('--heat', action='store_true', default=False,
                    help='plot heat map')
parser.add_argument('--var', action='store_true', default=False,
                    help='plot variance graph')
parser.add_argument('--bar', action='store_true', default=False,
                    help='plot bar chart')
parser.add_argument('--dataset', required=True,
                    help='which dataset')
parser.add_argument('--model', required=True,
                    help='which model')
parser.add_argument('--row_max',
                    help='which dataset', default=20)
parser.add_argument('--adv', action='store_true', default=False,
                    help='load adv model or clean model')
parser.add_argument('--attack', required=True,
                    help='which attack')
parser.add_argument('--samplesize',
                    help='sample size', default=1000)
parser.add_argument('--round',
                    help='the current round', default=1)
parser.add_argument('--num_data',
                    help='number of data for retraining', default=1)
parser.add_argument('--model_path',
                    help='path of model to be evaluated', default=None)
parser.add_argument('--train',
                    help='training data', default=False)

torch.manual_seed(1)
args = parser.parse_args()
attack = args.attack
adv = "adv" if args.adv else "clean"
# settings
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

dataset_map = {
    "mnist":"datasets.MNIST(root='../data', train=" + str(args.train) + ", download=True, transform=transform_test)",
    "cifar10":"datasets.CIFAR10(root='../data', train=" + str(args.train) + ", download=True, transform=transform_test)",
    "driving":"torch.load(data_attack/driving/)",
    "drebin":"torch.load(data_attack/drebin/)",
    "PDF":"DatasetFromCSV('../data/PDF/train.csv')",
    "Drebin":"DatasetFromDrebin('../data/Drebin/')"
}

transform_test = transforms.Compose([transforms.ToTensor(),])
# if args.dataset in ["mnist","cifar10"]:
testset = eval(dataset_map[args.dataset])
# testset = eval("datasets." + args.dataset.upper() + "(root='../data', train=False, download=True, transform=transform_test)")
advset = None
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
if not args.gen:#does not need this, advset won't be used in gen
    if args.attack == "rand":
        transform_test = transforms.Compose([transforms.ToTensor(), noise.AddGaussianNoise(0.0, args.epsilon)])
        advset = eval(dataset_map[args.dataset])
        # advset = eval("datasets." + args.dataset.upper() + "(root='../data', train=False, download=True, transform=transform_test)")
    elif args.attack == "pgd":
        # print("Now is pgd attack")
        # advset = torch.load("data_attack/" + args.dataset + "_" + adv + "_" + str(args.epsilon) + ".pt")
        # advset = torch.load("data_attack/" + args.dataset + "_" + args.model + "_" + adv + "_" + str(args.epsilon)  + ".pt")
        advset = torch.load("data_attack/" + args.dataset + "_" + adv + "_" + str(args.epsilon) + ".pt")
        # print("nature of advset: " + str(advset))
        # print("example advset: " + str(advset[0]))
    else:
        assert False, "invalid attack"

    if args.retrain:
        len_dataset = len(advset)
        remaining_dataset = len_dataset - int(args.num_data)
        this_sample,non_sample = torch.utils.data.dataset.random_split(advset,[int(args.num_data),remaining_dataset])
        print("len_dataset: " + str(len_dataset))#debug
        print("len this_sample: " + str(len(this_sample)))#debug
        adv_loader = torch.utils.data.DataLoader(this_sample, batch_size=args.test_batch_size, shuffle=True, num_workers=0)
    else:
        adv_loader = torch.utils.data.DataLoader(advset,batch_size=args.test_batch_size, shuffle=True, num_workers=0)
    # print("number of data: " + str(len(advset)))
    # for data, label in adv_loader:
    #     print("data:" + str(data))
    #     assert False, "debug"
    # for x in adv_loader:
    #     x = x.to('cuda', non_blocking=True)

# def load_h5_model(model_path):
#     try:
#         json_file = open(model_path + '.json', 'r')  # Read Keras model parameters (stored in JSON file)
#         file_content = json_file.read()
#         json_file.close()
#
#         model = model_from_json(file_content)
#         model.load_weights(model_path + '.h5')
#
#         # Compile the model before using
#         model.compile(loss='categorical_crossentropy',
#                       optimizer='adam',
#                       metrics=['accuracy'])
#
#     except:
#         model = load_model(model_path + '.h5')

def main():
    if args.model_path == None:
        with open(r'yaml/model.yaml') as file:
            model_pth = yaml.load(file)[args.model]
    else:
        model_pth = args.model_path
    data_mode = {
        "wideresnet": "WideResNet().to(device)",#CIFAR
        "smallcnn": "SmallCNN().to(device)",#MNIST, sub lenet1
        "densenet121": "densenet121().to(device)",#cifar10
        "fcnet5": "FCNet5().to(device)",#MNIST
        "fcnet10": "FCNet10().to(device)",#MNIST
        "conv1dnet": "Conv1DNet().to(device)",#MNIST, sub lenet3
        "conv2dnet": "Conv2DNet().to(device)",#MNIST, sub lenet4
        "resnet56": "resnet56().to(device)",
        "PDF": "PDF().to(device)",
        "Drebin": "Drebin().to(device)"
    }
    print('pgd black-box attack')
    model_target = eval(data_mode[args.model])
    model_source = None
    model_target.load_state_dict(torch.load(model_pth, map_location=device))
        # load_h5_model(model_pth[args.model])
        # print("finish and return")
        # return
    # else:
    #     assert False, "please enter a valid model"
    # model_source.load_state_dict(torch.load(model_pth[args.model]["adv" if args.adv else "clean"], map_location=device))
    pgd_o = pgd_a.pgd(args,device,advset,testset,attack, args.epsilon, args.samplesize, args.row_max,args.dataset,args.model,adv,args.round)
    if args.eval:
        pgd_o.eval_acc(model_target, test_loader, adv_loader, device)
    elif args.retrain:
        tt_o = tt.retrain_o(args.model,model_target,args.dataset,device,adv_loader, attack, int(args.num_data),float(args.epsilon))
        tt_o.start_train()
    else:
        pgd_o.eval_adv_test_blackbox(model_target, model_source, test_loader)

if __name__ == '__main__':
    main()