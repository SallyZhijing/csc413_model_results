import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import cv2
from PIL import Image

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        if grad_output is not None and len(grad_output) > 0:
            self.gradients = grad_output[0]
        else:
            print("Backward hook did not receive gradients.")

    def generate_heatmap(self, input_tensor, class_idx):
        # Ensure gradients are computed for the input
        input_tensor.requires_grad_()

        # Forward pass
        output = self.model(input_tensor)
        if class_idx >= output.size(1):
            raise ValueError(f"class_idx {class_idx} is out of range for model output size {output.size(1)}.")
        target = output[0, class_idx]

        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)

        # Grad-CAM calculation
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations are not captured. Check hooks and backward pass.")
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.nn.functional.relu(heatmap)
        heatmap /= torch.max(heatmap)
        return heatmap.detach().cpu().numpy()

# Load CIFAR-10 dataset (Cats and Dogs only)
class_to_include = [3, 5]  # 3 = Cat, 5 = Dog

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=preprocess)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=preprocess)

# Filter the dataset to include only Cats and Dogs
filtered_train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in class_to_include]
filtered_train_dataset = Subset(train_dataset, filtered_train_indices)

filtered_test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in class_to_include]
filtered_test_dataset = Subset(test_dataset, filtered_test_indices)

# Create DataLoaders
train_loader = DataLoader(filtered_train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(filtered_test_dataset, batch_size=64, shuffle=False)

# Load the pretrained model
model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

# Freeze all layers except the last two and the classifier
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# Modify the classifier for binary classification
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Get a different image from the test dataset
image, label = filtered_test_dataset[10]  # Choose the 11th image
input_tensor = image.unsqueeze(0).to('cuda')  # Add batch dimension

# Save the input image
input_image_path = 'input_image.jpg'
image_np = np.transpose(image.cpu().numpy(), (1, 2, 0))  # Convert CHW to HWC
image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))  # De-normalize
image_np = np.clip(image_np, 0, 1)  # Clip to valid range
image_np = (image_np * 255).astype(np.uint8)  # Convert to uint8
cv2.imwrite(input_image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
print(f"Original input image saved to {input_image_path}")

# Save the resized image for better readability
resized_image_path = 'resized_input_image.jpg'
resized_image = cv2.resize(image_np, (224, 224))
cv2.imwrite(resized_image_path, resized_image)
print(f"Resized input image saved to {resized_image_path}")

# Grad-CAM layers to visualize
layers_to_visualize = {
    'layer1': model.layer1,
    'layer2': model.layer2,
    'layer3': model.layer3,
    'layer4': model.layer4
}

# Generate Grad-CAM heatmaps for each layer
for layer_name, layer in layers_to_visualize.items():
    grad_cam = GradCAM(model, layer)  # Use specified layer

    # Generate Grad-CAM heatmap
    class_idx = class_to_include.index(label)  # Map label to class index (0 for Cat, 1 for Dog)
    heatmap = grad_cam.generate_heatmap(input_tensor, class_idx)

    # Overlay heatmap on the original image
    heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)

    # Save the heatmap
    output_path = f'gradcam_output_{layer_name}.jpg'
    cv2.imwrite(output_path, superimposed_img)
    print(f"Grad-CAM heatmap for {layer_name} saved to {output_path}")

