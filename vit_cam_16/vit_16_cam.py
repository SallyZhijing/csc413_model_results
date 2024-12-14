# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# from PIL import Image
# import os

# # Create output directory
# output_dir = "./gradcam_results"
# os.makedirs(output_dir, exist_ok=True)

# # -----------------------------------
# # Load Input Image (input_image.jpg)
# # -----------------------------------
# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=(0.4914, 0.4822, 0.4465),
#         std=(0.2470, 0.2435, 0.2616)
#     )
# ])

# # Load the image
# image_path = './input_image.jpg'
# image = Image.open(image_path).convert('RGB')
# image = preprocess(image)
# input_image = image.unsqueeze(0)

# # -----------------------------------
# # Load Pretrained ViT Model
# # -----------------------------------
# model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
# model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# input_image = input_image.to(device)

# # -----------------------------------
# # Grad-CAM class
# # -----------------------------------
# class ViTGradCAM:
#     def __init__(self, model, block_index):
#         self.model = model
#         self.block_index = block_index
#         self.activations = None
#         self.gradients = None
#         self.hook_handles = []
#         self._register_hooks()

#     def _get_target_layer(self):
#         last_block = self.model.encoder.layers[self.block_index]
#         return last_block.mlp[3]

#     def _forward_hook(self, module, input, output):
#         self.activations = output.detach()

#     def _backward_hook(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0].detach()

#     def _register_hooks(self):
#         target_layer = self._get_target_layer()
#         fh = target_layer.register_forward_hook(self._forward_hook)
#         bh = target_layer.register_full_backward_hook(self._backward_hook)
#         self.hook_handles.append(fh)
#         self.hook_handles.append(bh)

#     def remove_hooks(self):
#         for handle in self.hook_handles:
#             handle.remove()

#     def __call__(self, input_tensor, target_class):
#         output = self.model(input_tensor)
#         self.model.zero_grad()
#         one_hot = torch.zeros_like(output)
#         one_hot[0, target_class] = 1
#         output.backward(gradient=one_hot, retain_graph=True)

#         # Normalize gradients for better visualization
#         grads = self.gradients.mean(dim=-1, keepdim=True)
#         grads = (grads - grads.min()) / (grads.max() + 1e-8)

#         patch_activations = self.activations[0, 1:]
#         patch_weights = grads[0, 1:]

#         cam = (patch_weights * patch_activations).sum(dim=1).cpu().numpy()
#         cam = cam.reshape(14, 14)
#         cam = (cam - cam.min()) / (cam.max() + 1e-8)

#         return cam

# # -----------------------------------
# # Determine the predicted class once
# # -----------------------------------
# model.zero_grad()
# output = model(input_image)
# pred_class = output.argmax(dim=1).item()

# # -----------------------------------
# # Compute Grad-CAM for each encoder layer
# # -----------------------------------
# num_layers = len(model.encoder.layers)
# all_cams = []

# for i in range(num_layers):
#     gradcam = ViTGradCAM(model, block_index=i)
#     cam = gradcam(input_image, target_class=pred_class)
#     gradcam.remove_hooks()
#     all_cams.append(cam)

# # -----------------------------------
# # Visualization: Show and Save Heatmaps
# # -----------------------------------
# # Inverse normalization to show original image
# inv_normalize = transforms.Normalize(
#     mean=(-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616),
#     std=(1/0.2470, 1/0.2435, 1/0.2616)
# )
# img_show = inv_normalize(image).permute(1, 2, 0).cpu().numpy()
# img_show = np.clip(img_show, 0, 1)

# fig, axes = plt.subplots(3, 4, figsize=(12, 9))  # for 12 layers (3 rows x 4 cols)
# axes = axes.flatten()

# for idx, cam in enumerate(all_cams):
#     # Normalize CAM more precisely
#     cam_resized = cv2.resize(cam, (224, 224))
#     cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() + 1e-8)  # Re-normalize

#     # Generate heatmap
#     heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

#     # Blend heatmap and original image for overlay
#     heatmap = np.float32(heatmap) / 255.0
#     overlay = 0.5 * heatmap + 0.5 * img_show  # Adjust transparency

#     # Save each block's image separately
#     overlay_image = (overlay * 255).astype(np.uint8)
#     overlay_image = Image.fromarray((overlay_image * 255).astype(np.uint8))
#     overlay_image.save(f"{output_dir}/block_{idx}.png")

#     # Display the result
#     axes[idx].imshow(overlay)
#     axes[idx].set_title(f"Block {idx}")
#     axes[idx].axis('off')

# # Hide extra subplots if any
# for j in range(num_layers, len(axes)):
#     axes[j].axis('off')

# # Save the combined visualization as one image
# combined_path = f"{output_dir}/all_blocks.png"
# plt.tight_layout()
# plt.savefig(combined_path)

# plt.show()

# print("Predicted class:", pred_class)
import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os

# Create output directory
output_dir = "./gradcam_results"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------
# Load Input Image (input_image.jpg)
# -----------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

# Load the image
image_path = './input_image.jpg'
image = Image.open(image_path).convert('RGB')
image = preprocess(image)
input_image = image.unsqueeze(0)

# -----------------------------------
# Load Pretrained ViT Model
# -----------------------------------
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_image = input_image.to(device)

# -----------------------------------
# Grad-CAM class
# -----------------------------------
class ViTGradCAM:
    def __init__(self, model, block_index):
        self.model = model
        self.block_index = block_index
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _get_target_layer(self):
        last_block = self.model.encoder.layers[self.block_index]
        return last_block.mlp[3]

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def _register_hooks(self):
        target_layer = self._get_target_layer()
        fh = target_layer.register_forward_hook(self._forward_hook)
        bh = target_layer.register_full_backward_hook(self._backward_hook)
        self.hook_handles.append(fh)
        self.hook_handles.append(bh)

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_tensor, target_class):
        output = self.model(input_tensor)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Normalize gradients for better visualization
        grads = self.gradients.mean(dim=-1, keepdim=True)
        grads = (grads - grads.min()) / (grads.max() + 1e-8)

        patch_activations = self.activations[0, 1:]
        patch_weights = grads[0, 1:]

        cam = (patch_weights * patch_activations).sum(dim=1).cpu().numpy()
        cam = cam.reshape(14, 14)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam

# -----------------------------------
# Determine the predicted class once
# -----------------------------------
model.zero_grad()
output = model(input_image)
pred_class = output.argmax(dim=1).item()

# -----------------------------------
# Compute Grad-CAM for each encoder layer
# -----------------------------------
num_layers = len(model.encoder.layers)
all_cams = []

for i in range(num_layers):
    gradcam = ViTGradCAM(model, block_index=i)
    cam = gradcam(input_image, target_class=pred_class)
    gradcam.remove_hooks()
    all_cams.append(cam)

# -----------------------------------
# Visualization: Show and Save Heatmaps
# -----------------------------------
# Inverse normalization to show original image
inv_normalize = transforms.Normalize(
    mean=(-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616),
    std=(1/0.2470, 1/0.2435, 1/0.2616)
)
img_show = inv_normalize(image).permute(1, 2, 0).cpu().numpy()
img_show = np.clip(img_show, 0, 1)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))  # for 12 layers (3 rows x 4 cols)
axes = axes.flatten()

for idx, cam in enumerate(all_cams):
    # Normalize CAM more precisely
    cam_resized = cv2.resize(cam, (224, 224))
    cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() + 1e-8)  # Re-normalize

    # Generate heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    # Blend heatmap and original image for overlay
    heatmap = np.float32(heatmap) / 255.0
    overlay = 0.5 * heatmap + 0.5 * img_show  # Adjust transparency

    # Ensure consistent processing for individual images
    overlay_image = (overlay * 255).astype(np.uint8)
    overlay_image_pil = Image.fromarray(overlay_image)
    overlay_image_pil.save(f"{output_dir}/block_{idx}.png")

    # Display the result
    axes[idx].imshow(overlay)
    axes[idx].set_title(f"Block {idx}")
    axes[idx].axis('off')

# Hide extra subplots if any
for j in range(num_layers, len(axes)):
    axes[j].axis('off')

# Save the combined visualization as one image
combined_path = f"{output_dir}/all_blocks.png"
plt.tight_layout()
plt.savefig(combined_path)

plt.show()

print("Predicted class:", pred_class)
