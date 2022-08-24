import torch

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def random_attack(dataset, mean=0.0, sd=1.0):
    transform_test = transforms.Compose([transforms.ToTensor(), AddGaussianNoise(mean, sd)])
    if dataset == "cifar":
        print("load cifar dataset")
        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    elif dataset == "mnist":
        print("load mnist dataset")
        testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)
    return testset
