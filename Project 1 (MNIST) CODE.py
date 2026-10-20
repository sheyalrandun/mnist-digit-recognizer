cd


transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root = "data",train = True,download = True,transform = transform)
test_dataset = datasets.MNIST(root = "data",train = False, download = True, transform = transform)

train_base,val_base = random_split(train_dataset,[54000,6000],generator=torch.Generator().manual_seed(42))

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
        x = torch.flatten(x,start_dim = 1)
        x = self.classifier(x)
        return x

torch.manual_seed(42)
model = Project()

train_loader = DataLoader(train_base,batch_size = 64,shuffle = True)
val_loader = DataLoader(val_base,batch_size = 64,shuffle = False)
test_loader = DataLoader(test_dataset,batch_size = 64,shuffle = False)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

training_loss = []
validation_loss = []
validation_accuracy = []
training_accuracy = []

for i in range(10):
    model.train()
    train_count = 0
    temp_train_loss = 0
    correct = 0
    for images,labels in train_loader:
        optimizer.zero_grad()
        train_output = model(images)
        train_loss = loss(train_output,labels)
        temp_train_loss += train_loss.item()*images.size(0)
        train_count += 1
        prediction = train_output.argmax(dim = 1)
        correct += ((prediction == labels).sum().item())
        train_loss.backward()
        optimizer.step()

    training_accuracy.append((correct / len(train_base))*100)

    model.eval()
    with torch.no_grad():
        val_count = 0
        temp_val_loss = 0
        correct = 0
        for images,labels in val_loader:
            val_output = model(images)
            val_loss = loss(val_output,labels)
            temp_val_loss += val_loss.item()*images.size(0)
            val_count += 1
            predictions = val_output.argmax(dim = 1)
            correct += (predictions == labels).sum().item()

    validation_accuracy.append((correct / len(val_base))*100)
    training_loss.append(temp_train_loss/len(train_base))
    validation_loss.append(temp_val_loss/len(val_base))

print(f"Training Loss :{training_loss}")
print(f"Validation Loss :{validation_loss}")
print(f"Validation Accuracy :{validation_accuracy}")
print(f"Training Accuracy :{training_accuracy}")

torch.save(model.state_dict(),"mnist_model_10epoch.pth")
print("Saved successfully")