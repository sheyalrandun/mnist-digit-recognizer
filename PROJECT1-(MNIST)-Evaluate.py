import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets,transforms

class Project(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,8,kernel_size = 3,padding = 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8,16,kernel_size = 3,padding = 1)
        self.classifier = nn.Linear(784,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.classifier(x)
        return x

transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root = "data",train = False,download = False,transform = transform)
test_loader = DataLoader(test_dataset,batch_size = 64,shuffle = False)

model = Project()
model.load_state_dict(torch.load("mnist_model_10epoch.pth"))

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

test_accuracy = (correct / total) * 100
print(f"Test accuracy :{test_accuracy:.2f}%")
