import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import copy
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# Hyperparameters
dataset_size = 1
batch_size = 32
epochs = 10
learning_rate = 0.01

# Training function
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_history = []
    train_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                scheduler.step()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # We record training metrics for the 'train' phase
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())

            # Update best model if we are in 'val' phase and have a better accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Train Loss: {train_loss_history[-1]:.4f}, Train Acc: {train_acc_history[-1]:.4f}")

    model.load_state_dict(best_model_wts)
    return model, train_loss_history, train_acc_history

# Evaluate function
def evaluate_model(model, dataloader, dataset_size, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    loss = running_loss / dataset_size
    acc = running_corrects.double() / dataset_size
    return loss, acc

# Now we use the muffin_chihuahua dataset
dataset_name = "muffin_chihuahua"
print(f"Training on dataset: {dataset_name}")

full_dataset = datasets.ImageFolder(root=dataset_name, transform=data_transforms['train'])
train_size = int(0.6 * len(full_dataset) * dataset_size)
val_size = int(0.1 * len(full_dataset) * dataset_size)
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
val_dataset.dataset.transform = data_transforms['val']
test_dataset.dataset.transform = data_transforms['test']

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
}
dataset_sizes = {'train': train_size, 'val': val_size, 'test': test_size}

# Initialize model - pretrained ResNet18
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(full_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
model, train_loss_history, train_acc_history = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=epochs)

# Save the model
model_path = f"resnet18_pretrained_{dataset_name}.pth"
torch.save(model.state_dict(), model_path)

# Evaluate on the test set
test_loss, test_acc = evaluate_model(model, dataloaders['test'], dataset_sizes['test'], criterion)
print(f"Test Loss on {dataset_name}: {test_loss}, Test Acc: {test_acc:.4f}")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(range(len(train_loss_history)), train_loss_history, label=f'{dataset_name} Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'resnet_18_pretrained_{dataset_name}_loss.png')
plt.show()

# Plot training accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(len(train_acc_history)), train_acc_history, label=f'{dataset_name} Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'resnet_18_pretrained_{dataset_name}_accuracy.png')
plt.show()

# Output saved model path
print("Saved model path:")
print(f"{dataset_name}: {model_path}")
