import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
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
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        # Forward pass
        output = self.model(input_tensor)
        target = output[0, class_idx]

        # Backward pass
        self.model.zero_grad()
        target.backward()

        # Grad-CAM calculation
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.nn.functional.relu(heatmap)
        heatmap /= torch.max(heatmap)
        return heatmap.detach().cpu().numpy()

# Load the trained model
model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)  # Modify for binary classification
model.load_state_dict(torch.load('best_model.pth'))  # Load your trained model
model.eval()

# Choose the layer to visualize
grad_cam = GradCAM(model, model.layer4)  # Use ResNet's `layer4`

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load CIFAR-10 dataset (Cats and Dogs only)
class_to_include = [3, 5]  # 3 = Cat, 5 = Dog

# CIFAR-10 test dataset
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=preprocess)

# Filter the dataset to include only Cats and Dogs
filtered_indices = [i for i, (_, label) in enumerate(test_dataset) if label in class_to_include]
filtered_dataset = torch.utils.data.Subset(test_dataset, filtered_indices)

# Get a single image from the filtered dataset
image, label = filtered_dataset[0]  # Choose the first image
input_tensor = image.unsqueeze(0)  # Add batch dimension

# Generate Grad-CAM heatmap
class_idx = class_to_include.index(label)  # Map label to class index (0 for Cat, 1 for Dog)
heatmap = grad_cam.generate_heatmap(input_tensor, class_idx)

# Convert tensor to image for overlay
image_np = np.transpose(image.numpy(), (1, 2, 0))  # Convert CHW to HWC
image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))  # De-normalize
image_np = np.clip(image_np, 0, 1)  # Clip to valid range
image_np = (image_np * 255).astype(np.uint8)  # Convert to uint8

# Overlay heatmap on the original image
heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
heatmap_resized = np.uint8(255 * heatmap_resized)
heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)

# Save and display the result
output_path = 'gradcam_output.jpg'
cv2.imwrite(output_path, superimposed_img)
print(f"Grad-CAM heatmap saved to {output_path}")

