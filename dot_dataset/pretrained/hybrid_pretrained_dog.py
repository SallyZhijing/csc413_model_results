import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import copy
import time

import torch
import torch.nn as nn
from torchvision import models

class HybridCNNViT(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(HybridCNNViT, self).__init__()
        # A simple CNN feature extraction block
        self.cnn_features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # Now image is 8 x 112 x 112
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Back to 3 x 224 x 224
        )

        # Load ViT model
        self.vit = models.vit_b_16(pretrained=pretrained)
        # Replace the classification head
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        x = self.cnn_features(x)
        x = self.vit(x)
        return x


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

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    val_loss_history = []
    val_acc_history = []

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

            if phase == 'val':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Validation Loss: {val_loss_history[-1]:.4f}, Validation Acc: {val_acc_history[-1]:.4f}")

    model.load_state_dict(best_model_wts)
    return model, val_loss_history, val_acc_history

def evaluate_model(model, dataloader, dataset_size):
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

datasets_list = ["SDDsubset", "DBIsubset"]
loss_results = {}
acc_results = {}
model_paths = {}

criterion = nn.CrossEntropyLoss()

for pretrained_flag in [True, False]:
    for dataset_name in datasets_list:
        print(f"Training on dataset: {dataset_name} with ViT pretrained={pretrained_flag}")
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

        # Initialize model
        model = HybridCNNViT(num_classes=len(full_dataset.classes), pretrained=pretrained_flag)
        model = model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Train the model
        model, val_loss_history, val_acc_history = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=epochs)

        # Save the model
        model_path = f"best_hybrid_model_vit_b_16_pretrained{pretrained_flag}_{dataset_name}.pth"
        torch.save(model.state_dict(), model_path)
        model_paths[f"{dataset_name}_pretrained{pretrained_flag}"] = model_path

        # Evaluate on the test set
        test_loss, test_acc = evaluate_model(model, dataloaders['test'], dataset_sizes['test'])
        print(f"Test Loss on {dataset_name} (pretrained={pretrained_flag}): {test_loss}, Test Acc: {test_acc:.4f}")

        # Save validation loss and accuracy history
        loss_results[f"{dataset_name}_pretrained{pretrained_flag}"] = val_loss_history
        acc_results[f"{dataset_name}_pretrained{pretrained_flag}"] = val_acc_history

        # ----- Produce per-run plots for this model and dataset -----
        # Plot validation loss for this run
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(val_loss_history)+1), val_loss_history, label='Validation Loss')
        plt.title(f'Validation Loss for {dataset_name}, Pretrained={pretrained_flag}, Hybrid CNN+ViT')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'validation_loss_{dataset_name}_pretrained{pretrained_flag}_hybrid_cnn_vit.png')
        plt.close()

        # Plot validation accuracy for this run
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(val_acc_history)+1), val_acc_history, label='Validation Accuracy')
        plt.title(f'Validation Accuracy for {dataset_name}, Pretrained={pretrained_flag}, Hybrid CNN+ViT')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'validation_accuracy_{dataset_name}_pretrained{pretrained_flag}_hybrid_cnn_vit.png')
        plt.close()

# After all runs, produce comparison plots
plt.figure(figsize=(10, 5))
for key, loss_history in loss_results.items():
    plt.plot(range(len(loss_history)), loss_history, label=f'{key} Validation Loss')

plt.title('Validation Loss Comparison (Hybrid CNN+ViT)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('validation_loss_comparison_hybrid_cnn_vit.png')
plt.show()

plt.figure(figsize=(10, 5))
for key, acc_history in acc_results.items():
    plt.plot(range(len(acc_history)), acc_history, label=f'{key} Validation Accuracy')

plt.title('Validation Accuracy Comparison (Hybrid CNN+ViT)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('validation_accuracy_comparison_hybrid_cnn_vit.png')
plt.show()

# Output saved model paths
print("Saved model paths:")
for key, path in model_paths.items():
    print(f"{key}: {path}")
