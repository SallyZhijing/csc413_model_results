import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
from sklearn.metrics import precision_score, f1_score
import csv
import matplotlib.pyplot as plt
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],  # CIFAR-10 mean
                             [0.2470, 0.2435, 0.2616])  # CIFAR-10 std
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2470, 0.2435, 0.2616])
    ])
}

# Choose two classes (e.g., "cat" and "dog")
class_to_include = [3, 5]  # CIFAR-10 indices: 3=cat, 5=dog

def filter_dataset(dataset, class_indices):
    filtered_indices = [i for i, (_, label) in enumerate(dataset) if
                        label in class_indices]
    return Subset(dataset, filtered_indices)

# Load CIFAR-10 dataset
full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                      transform=data_transforms['train'])
full_val_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                    transform=data_transforms['val'])

# Filter datasets to include only the chosen classes
train_dataset_full = filter_dataset(full_train_dataset, class_to_include)
val_dataset = filter_dataset(full_val_dataset, class_to_include)

# Update class labels to be binary (0 and 1)
def relabel_dataset(subset):
    for i in range(len(subset)):
        img, label = subset[i]
        subset.dataset.targets[subset.indices[i]] = class_to_include.index(
            label)
    return subset

train_dataset_full = relabel_dataset(train_dataset_full)
val_dataset = relabel_dataset(val_dataset)

# Create subsets for 25%, 50%, and 100%
train_size_quarter = len(train_dataset_full) // 4
train_size_half = len(train_dataset_full) // 2

train_dataset_quarter = Subset(train_dataset_full.dataset,
                               train_dataset_full.indices[:train_size_quarter])
train_dataset_half = Subset(train_dataset_full.dataset,
                            train_dataset_full.indices[:train_size_half])

# train_dataset_full is already the 100%

def get_dataloaders(train_subset, val_dataset, batch_size=64):
    dataloaders = {
        'train': DataLoader(train_subset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    }
    dataset_sizes = {'train': len(train_subset), 'val': len(val_dataset)}
    return dataloaders, dataset_sizes

def initialize_model(unfreeze_layers=1):
    # Load pretrained ResNet101
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the specified layers
    if unfreeze_layers >= 1:
        for param in model.layer4.parameters():
            param.requires_grad = True
    if unfreeze_layers >= 2:
        for param in model.layer3.parameters():
            param.requires_grad = True
    if unfreeze_layers >= 3:
        for param in model.layer2.parameters():
            param.requires_grad = True

    # Always unfreeze the classifier (fc layer)
    for param in model.fc.parameters():
        param.requires_grad = True

    # Modify the classifier for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
    return model.to(device)

def compute_metrics(preds, labels):
    # Ensure inputs are 1D tensors
    preds = preds.view(-1)
    labels = labels.view(-1)

    # Convert tensors to numpy arrays for compatibility with sklearn
    preds_cpu = preds.cpu().numpy()
    labels_cpu = labels.cpu().numpy()

    # Calculate metrics
    accuracy = np.mean(preds_cpu == labels_cpu)
    precision = precision_score(labels_cpu, preds_cpu, average='binary')
    f1 = f1_score(labels_cpu, preds_cpu, average='binary')

    return accuracy, precision, f1

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes,
                num_epochs=10, run_label="50% Dataset"):
    results = []  # will store dicts per epoch

    for epoch in range(num_epochs):
        print(f'Run ({run_label}) - Epoch {epoch + 1}/{num_epochs}')
        epoch_results = {'run': run_label, 'epoch': epoch + 1}

        for phase in ['train', 'val']:
            start_time = time.time()
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_total = 0

            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data).item()
                running_total += labels.size(0)
                running_loss += loss.item() * inputs.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_accuracy = running_corrects / running_total

            # Compute additional metrics
            precision = precision_score(all_labels, all_preds, average='binary')
            f1 = f1_score(all_labels, all_preds, average='binary')

            elapsed_time = time.time() - start_time

            print(
                f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, Time: {elapsed_time:.2f}s")

            epoch_results[f'{phase}_loss'] = epoch_loss
            epoch_results[f'{phase}_acc'] = epoch_accuracy
            epoch_results[f'{phase}_precision'] = precision
            epoch_results[f'{phase}_f1'] = f1
            epoch_results[f'{phase}_time'] = elapsed_time

        results.append(epoch_results)

    return results

# Prepare criterion
criterion = nn.CrossEntropyLoss()

# Run 1: Only the last block (layer4) unfrozen
dataloaders_half, dataset_sizes_half = get_dataloaders(train_dataset_half,
                                                       val_dataset)
model = initialize_model(unfreeze_layers=1)
params_to_update = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params_to_update, lr=0.01, momentum=0.9)
results_layer4 = train_model(model, criterion, optimizer, dataloaders_half,
                              dataset_sizes_half, num_epochs=10,
                              run_label="50% Dataset (layer4)")

# Run 2: Last two blocks (layer3 and layer4) unfrozen
model = initialize_model(unfreeze_layers=2)
params_to_update = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params_to_update, lr=0.01, momentum=0.9)
results_layer3_4 = train_model(model, criterion, optimizer, dataloaders_half,
                                dataset_sizes_half, num_epochs=10,
                                run_label="50% Dataset (layer3, layer4)")

# Run 3: Last three blocks (layer2, layer3, and layer4) unfrozen
model = initialize_model(unfreeze_layers=3)
params_to_update = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params_to_update, lr=0.01, momentum=0.9)
results_layer2_3_4 = train_model(model, criterion, optimizer, dataloaders_half,
                                  dataset_sizes_half, num_epochs=10,
                                  run_label="50% Dataset (layer2, layer3, layer4)")

# Save results to CSV
csv_filename = "training_results.csv"
fieldnames = [
    'run', 'epoch',
    'train_loss', 'train_acc', 'train_precision', 'train_f1', 'train_time',
    'val_loss', 'val_acc', 'val_precision', 'val_f1', 'val_time'
]
with open(csv_filename, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results_layer4:
        writer.writerow(row)
    for row in results_layer3_4:
        writer.writerow(row)
    for row in results_layer2_3_4:
        writer.writerow(row)


print(f"Results saved to {csv_filename}")

# Plot accuracy for all three runs
epochs_layer4 = [r['epoch'] for r in results_layer4]
acc_layer4 = [r['val_acc'] for r in results_layer4]

epochs_layer3_4 = [r['epoch'] for r in results_layer3_4]
acc_layer3_4 = [r['val_acc'] for r in results_layer3_4]

epochs_layer2_3_4 = [r['epoch'] for r in results_layer2_3_4]
acc_layer2_3_4 = [r['val_acc'] for r in results_layer2_3_4]

plt.figure(figsize=(8, 6))
plt.plot(epochs_layer4, acc_layer4, label='Layer4 Unfrozen')
plt.plot(epochs_layer3_4, acc_layer3_4, label='Layer3-4 Unfrozen')
plt.plot(epochs_layer2_3_4, acc_layer2_3_4, label='Layer2-4 Unfrozen')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs. Epoch for Different Unfreeze Strategies')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')
plt.show()

