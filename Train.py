import torch
import matplotlib.pyplot as plt
import numpy as np

def train(net, train_loader, val_loader, val_num, epochs, optimizer, loss, device, save_path):
    net.to(device)
    best_acc = 0
    lossList = []
    accList = []
    x = np.linspace(0, (epochs - 1), epochs)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        lossEnd = 0.0
        for i, (X, y) in enumerate(train_loader):

            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            running_loss += l.item()
            rate = (i + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            lossEnd = running_loss / (i+1)
            print("\rEpoch: {}  {:^3.0f}%[{}->{}] train loss: {:.5f}".format(epoch, int(rate * 100), a, b,
                                                                             lossEnd), end="")
        lossList.append(lossEnd)
        print()

        # validation
        net.eval()
        acc = 0.0
        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data
                optimizer.zero_grad()
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]

                acc += (predict_y == val_labels.to(device)).sum().item()
            val_accurate = acc / val_num
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
            print('Epoch: {}   val_accuracy: {:.5f}'.format(epoch, val_accurate))
            accList.append(val_accurate)

    print('Finished Training')
    plt.plot(x, lossList, 'or-', label=r'train_loss')
    plt.plot(x, accList, 'ob-', label=r'val_accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('End.png')
    plt.show()