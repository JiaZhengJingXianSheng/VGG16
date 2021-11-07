import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
import torch
import model
import Train


data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

train_data_path = "Data"
val_data_path = "val"
batch_size = 8
num_workers = 8
device = torch.device("cuda:0")
epochs = 50
save_path = 'One_Piece.pth'
num_classes = 7

if __name__ == "__main__":
    train_data = datasets.ImageFolder(root=train_data_path, transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)

    val_data = datasets.ImageFolder(root=val_data_path, transform=data_transform["val"])
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers)

    net = model.vgg16(batch_norm=True, num_classes=num_classes, init_weights=True)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0005)
    Train.train(net, train_loader, val_loader, len(val_data), epochs, optimizer, loss, device, save_path)
