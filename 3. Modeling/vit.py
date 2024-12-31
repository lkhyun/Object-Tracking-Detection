import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
import timm
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
os.chdir("C:/Users/dlrkd/Desktop/hai-lab/model")

def load_data(file_path):
    full_data = np.load(file_path)
    data = full_data[:, :-1].reshape(-1, 1, 51, 360)  # Reshape
    labels = full_data[:, -1]
    
    # Normalize data(z-score)
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std == 0] = 1
    data = (data - mean) / std
    
    return data, labels

# Data preparation
all_data_com, all_labels_com = [], []
all_data_tv, all_labels_tv = [], []

for i in range(1, 49):
    data_com, labels_com = load_data(f"{i}_com.npy")
    data_tv, labels_tv = load_data(f"{i}_tv.npy")
    all_data_com.append(data_com)
    all_labels_com.append(labels_com)
    all_data_tv.append(data_tv)
    all_labels_tv.append(labels_tv)

# Convert to tensors
all_data_com = torch.tensor(np.concatenate(all_data_com), dtype=torch.float32)
all_labels_com = torch.tensor(np.concatenate(all_labels_com), dtype=torch.int64)
all_data_tv = torch.tensor(np.concatenate(all_data_tv), dtype=torch.float32)
all_labels_tv = torch.tensor(np.concatenate(all_labels_tv), dtype=torch.int64)

# Split the data into train, validation, and test sets
train_dataset_com = TensorDataset(all_data_com[:3193], all_labels_com[:3193])
valid_dataset_com = TensorDataset(all_data_com[3193:3613], all_labels_com[3193:3613])
test_dataset_com = TensorDataset(all_data_com[3613:4033], all_labels_com[3613:4033])

train_dataset_tv = TensorDataset(all_data_tv[:3193], all_labels_tv[:3193])
valid_dataset_tv = TensorDataset(all_data_tv[3193:3613], all_labels_tv[3193:3613])
test_dataset_tv = TensorDataset(all_data_tv[3613:4033], all_labels_tv[3613:4033])

# Define the image transformations
image_size = 224  # Size expected by the ViT model
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Create a wrapper for TensorDataset to apply transformations
class TransformTensorDataset(Dataset):
    def __init__(self, tensor_data, tensor_labels, transform=None):
        self.tensor_data = tensor_data
        self.tensor_labels = tensor_labels
        self.transform = transform

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        sample = self.tensor_data[idx].numpy().squeeze()  # Squeeze to remove the singleton dimension
        label = self.tensor_labels[idx]
        
        if self.transform:
            sample = Image.fromarray((sample * 255).astype(np.uint8))  # Convert to PIL Image
            sample = sample.convert('RGB')  # Convert to RGB
            sample = self.transform(sample)
        
        return sample, label



# Create transformed datasets
train_dataset_com_transformed = TransformTensorDataset(all_data_com[:3193], all_labels_com[:3193], transform=transform)
valid_dataset_com_transformed = TransformTensorDataset(all_data_com[3193:3613], all_labels_com[3193:3613], transform=transform)
test_dataset_com_transformed = TransformTensorDataset(all_data_com[3613:4033], all_labels_com[3613:4033], transform=transform)

train_dataset_tv_transformed = TransformTensorDataset(all_data_tv[:3193], all_labels_tv[:3193], transform=transform)
valid_dataset_tv_transformed = TransformTensorDataset(all_data_tv[3193:3613], all_labels_tv[3193:3613], transform=transform)
test_dataset_tv_transformed = TransformTensorDataset(all_data_tv[3613:4033], all_labels_tv[3613:4033], transform=transform)

# Create data loaders
train_loader_com = DataLoader(train_dataset_com_transformed, batch_size=16, shuffle=True)
valid_loader_com = DataLoader(valid_dataset_com_transformed, batch_size=16, shuffle=False)
test_loader_com = DataLoader(test_dataset_com_transformed, batch_size=16, shuffle=False)

train_loader_tv = DataLoader(train_dataset_tv_transformed, batch_size=16, shuffle=True)
valid_loader_tv = DataLoader(valid_dataset_tv_transformed, batch_size=16, shuffle=False)
test_loader_tv = DataLoader(test_dataset_tv_transformed, batch_size=16, shuffle=False)

# Load a pre-trained Vision Transformer model
model_com = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=9)
model_tv = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=7)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_com.to(device)
model_tv.to(device)

# Loss function and optimizer
criterion_com = nn.CrossEntropyLoss()
optimizer_com = optim.Adam(model_com.parameters(), lr=1e-4)
scheduler_com = lr_scheduler.StepLR(optimizer_com, step_size=7, gamma=0.1)
criterion_tv = nn.CrossEntropyLoss()
optimizer_tv = optim.Adam(model_tv.parameters(), lr=1e-4)
scheduler_tv = lr_scheduler.StepLR(optimizer_tv, step_size=7, gamma=0.1)

num_epochs = 10  # Adjust based on your needs

# Training and validation loop
for epoch in range(num_epochs):
    model_com.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader_com):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer_com.zero_grad()
        outputs = model_com(inputs)
        loss = criterion_com(outputs, labels)
        loss.backward()
        optimizer_com.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler_com.step()

    epoch_loss = running_loss / len(train_loader_com.dataset)
    epoch_acc = correct / total

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Validate the model
    model_com.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader_com):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_com(inputs)
            loss = criterion_com(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(valid_loader_com.dataset)
    val_acc = val_correct / val_total

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# Evaluate on test data
model_com.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for inputs, labels in tqdm(test_loader_com):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_com(inputs)
        loss = criterion_com(outputs, labels)

        test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_loss /= len(test_loader_com.dataset)
test_acc = test_correct / test_total

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Repeat the same for TV datasets
# Training and validation loop for TV dataset
for epoch in range(num_epochs):
    model_tv.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader_tv):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer_tv.zero_grad()
        outputs = model_tv(inputs)
        loss = criterion_tv(outputs, labels)
        loss.backward()
        optimizer_tv.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler_tv.step()

    epoch_loss = running_loss / len(train_loader_tv.dataset)
    epoch_acc = correct / total

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Validate the model
    model_tv.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader_tv):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_tv(inputs)
            loss = criterion_tv(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(valid_loader_tv.dataset)
    val_acc = val_correct / val_total

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# Evaluate on test data for TV dataset
model_tv.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for inputs, labels in tqdm(test_loader_tv):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_tv(inputs)
        loss = criterion_tv(outputs, labels)

        test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_loss /= len(test_loader_tv.dataset)
test_acc = test_correct / test_total

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
