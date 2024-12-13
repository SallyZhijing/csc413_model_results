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
# Resize CIFAR-10 images to 224x224 for compatibility with ResNet pretrained on ImageNet
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],  # CIFAR-10 mean
                             [0.2470, 0.2435, 0.2616])  # CIFAR-10 std
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
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

def initialize_model():
    # Load pretrained ResNet152
    model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer3 and layer4
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
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
                num_epochs=10, run_label="Dataset"):
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

# Run 1: 25% dataset
dataloaders_quarter, dataset_sizes_quarter = get_dataloaders(train_dataset_quarter,
                                                             val_dataset)
model = initialize_model()
params_to_update = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params_to_update, lr=0.01, momentum=0.9)
results_quarter = train_model(model, criterion, optimizer, dataloaders_quarter,
                               dataset_sizes_quarter, num_epochs=10,
                               run_label="25% Dataset")

# Run 2: 50% dataset
dataloaders_half, dataset_sizes_half = get_dataloaders(train_dataset_half,
                                                       val_dataset)
model = initialize_model()
params_to_update = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params_to_update, lr=0.01, momentum=0.9)
results_half = train_model(model, criterion, optimizer, dataloaders_half,
                           dataset_sizes_half, num_epochs=10,
                           run_label="50% Dataset")

# Run 3: 100% dataset
dataloaders_full, dataset_sizes_full = get_dataloaders(train_dataset_full,
                                                       val_dataset)
model = initialize_model()
params_to_update = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params_to_update, lr=0.01, momentum=0.9)
results_full = train_model(model, criterion, optimizer, dataloaders_full,
                           dataset_sizes_full, num_epochs=10,
                           run_label="100% Dataset")

# Save results to CSV
csv_filename = "training_results_224.csv"
fieldnames = [
    'run', 'epoch',
    'train_loss', 'train_acc', 'train_precision', 'train_f1', 'train_time',
    'val_loss', 'val_acc', 'val_precision', 'val_f1', 'val_time'
]
with open(csv_filename, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results_quarter:
        writer.writerow(row)
    for row in results_half:
        writer.writerow(row)
    for row in results_full:
        writer.writerow(row)

print(f"Results saved to {csv_filename}")

# Plot metrics for all three runs
def plot_metric(metric_key, ylabel, title, filename):
    epochs_quarter = [r['epoch'] for r in results_quarter]
    metric_quarter = [r[metric_key] for r in results_quarter]

    epochs_half = [r['epoch'] for r in results_half]
    metric_half = [r[metric_key] for r in results_half]

    epochs_full = [r['epoch'] for r in results_full]
    metric_full = [r[metric_key] for r in results_full]

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_quarter, metric_quarter, label='25% Dataset')
    plt.plot(epochs_half, metric_half, label='50% Dataset')
    plt.plot(epochs_full, metric_full, label='100% Dataset')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# Plot validation accuracy
plot_metric('val_acc', 'Validation Accuracy', 'Validation Accuracy vs. Epoch for 224x224 CIFAR-10', 'accuracy_plot_224.png')

# Plot validation loss
plot_metric('val_loss', 'Validation Loss', 'Validation Loss vs. Epoch for 224x224 CIFAR-10', 'loss_plot_224.png')

# Plot validation precision
plot_metric('val_precision', 'Validation Precision', 'Validation Precision vs. Epoch for 224x224 CIFAR-10', 'precision_plot_224.png')

# Plot validation F1-score
plot_metric('val_f1', 'Validation F1 Score', 'Validation F1 Score vs. Epoch for 224x224 CIFAR-10', 'f1_plot_224.png')

