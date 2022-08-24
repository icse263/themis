from models.small_cnn import SmallCNN
import torch

device = torch.device("cpu")
model = SmallCNN().to(device)
model.load_state_dict(torch.load('./checkpoints/model_mnist_smallcnn.pt', map_location=device))
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

print(model.classifier[0].weight)
# for layer in model.classifier.modules():
#         print(layer.weight)