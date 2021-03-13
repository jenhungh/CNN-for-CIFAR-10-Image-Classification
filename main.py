import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self. conv_layer_set = nn.Sequential(
            # first conv block 
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p = 0.1),

            # second conv block
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p = 0.1),

            # third conv block
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p = 0.1),
        )

        self. fc_layer_set = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace = True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace = True),
            nn.Linear(512, 10),
        )


    def forward(self, x):
        # Convolution layer block
        x = self.conv_layer_set(x)

        x = x.view(-1, 256 * 4 * 4)

        # Fully-connected layer block
        x = self.fc_layer_set(x)

        return x


def main():
    # choose the device to use
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    # Set up the transform(Normalizaiton and Data Augmentation)
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),  # Crop
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),                   # Change shear angles and Zoom 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),    # Change brightness, contrast, and saturaiton  
        transforms.RandomHorizontalFlip(),                                       # Flip the image horizontally
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])   # Normalization

    transform_test = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])   # Normalization

    # load training datatset, test dataset, and classes
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=200,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=200,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Create the CNN model
    model = CNN()
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    torch.backends.cudnn.benchmark = True

    # Set up the criterion, optimizer, and learing rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.1)

    # Initialize the training loss history
    train_loss_history = [] 
    
    # Train the model
    num_epoch = 100
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        batch_loss = 0.0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute statistics
            batch_loss += loss.item()
            running_loss += loss.item()
            _, results = torch.max(outputs.data, 1)
            accuracy = results.eq(labels).float().mean()

            # Print training loss every 50 mini-batches
            if i % 50 == 49:
                print('[%d, %5d] loss: %.5f' %(epoch + 1, i + 1, batch_loss / 50))
                batch_loss = 0.0
            
        # Update lr if needed
        scheduler.step()
        
        # Update train_loss_history per epoch
        train_loss_history.append(running_loss / len(trainloader))

        # Print the accuracy per epoch
        print('epoch: %d, training accuracy: %.1f %%' %(epoch + 1, 100 * accuracy))

    # Finish training 
    print('Finished Training')

    # Plot the training process
    plt.plot(range(1, num_epoch+1), train_loss_history, 'r-')
    plt.ylim([0.0, 2.0])
    plt.title('Loss during training')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    # Save the model
    PATH = './model.pkl'
    torch.save(model.state_dict(), PATH)

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))

           
if __name__ == "__main__":
    main()