import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
import timm
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from PIL import Image
import os

os.chdir("C:/Users/dlrkd/Desktop/hai-lab/model")

def load_data(file_path):
    full_data = np.load(file_path)
    data = full_data[:, :-1].reshape(-1, 1, 72, 51)  # Reshape
    labels = full_data[:, -1]
    
    # # Normalize data(z-score)
    # mean = data.mean(axis=0, keepdims=True)
    # std = data.std(axis=0, keepdims=True)
    # std[std == 0] = 1
    # data = (data - mean) / std
    
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

# Groups for group k-fold
groups_com = np.concatenate([[i] * 14 for i in range(1, 289)])
groups_tv = np.concatenate([[i] * 14 for i in range(1, 289)])

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

# Training and validation loop
def train_and_evaluate(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Validate the model
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(valid_loader.dataset)
        val_acc = val_correct / val_total

        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# Evaluate on test data
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    total_loss /= len(data_loader.dataset)
    accuracy = correct / total

    print(f"Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
    return total_loss, accuracy

# GroupKFold cross-validation
group_kfold = GroupKFold(n_splits=5)

# Cross-validation for COM data
com_results = []
for fold, (train_idx, valid_idx) in enumerate(group_kfold.split(all_data_com, all_labels_com, groups=groups_com)):
    print(f"Fold {fold + 1}")
    
    # Prepare models, criteria, optimizers, and schedulers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_com = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=9).to(device)
    criterion_com = nn.CrossEntropyLoss()
    optimizer_com = optim.Adam(model_com.parameters(), lr=1e-4)
    scheduler_com = lr_scheduler.StepLR(optimizer_com, step_size=7, gamma=0.1)
    
    train_dataset_com = TransformTensorDataset(all_data_com[train_idx], all_labels_com[train_idx], transform=transform)
    valid_dataset_com = TransformTensorDataset(all_data_com[valid_idx], all_labels_com[valid_idx], transform=transform)

    train_loader_com = DataLoader(train_dataset_com, batch_size=16, shuffle=True)
    valid_loader_com = DataLoader(valid_dataset_com, batch_size=16, shuffle=False)

    train_and_evaluate(model_com, train_loader_com, valid_loader_com, criterion_com, optimizer_com, scheduler_com, device)
    
    loss, accuracy = evaluate(model_com, valid_loader_com, criterion_com, device)
    com_results.append((loss, accuracy))

# Cross-validation for TV data
tv_results = []
for fold, (train_idx, valid_idx) in enumerate(group_kfold.split(all_data_tv, all_labels_tv, groups=groups_tv)):
    print(f"Fold {fold + 1}")
    
    # Prepare models, criteria, optimizers, and schedulers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_tv = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=7).to(device)
    criterion_tv = nn.CrossEntropyLoss()
    optimizer_tv = optim.Adam(model_tv.parameters(), lr=1e-4)
    scheduler_tv = lr_scheduler.StepLR(optimizer_tv, step_size=7, gamma=0.1)
    
    train_dataset_tv = TransformTensorDataset(all_data_tv[train_idx], all_labels_tv[train_idx], transform=transform)
    valid_dataset_tv = TransformTensorDataset(all_data_tv[valid_idx], all_labels_tv[valid_idx], transform=transform)

    train_loader_tv = DataLoader(train_dataset_tv, batch_size=16, shuffle=True)
    valid_loader_tv = DataLoader(valid_dataset_tv, batch_size=16, shuffle=False)

    train_and_evaluate(model_tv, train_loader_tv, valid_loader_tv, criterion_tv, optimizer_tv, scheduler_tv, device)
    
    loss, accuracy = evaluate(model_tv, valid_loader_tv, criterion_tv, device)
    tv_results.append((loss, accuracy))

# Print cross-validation results
com_avg_loss = np.mean([result[0] for result in com_results])
com_avg_acc = np.mean([result[1] for result in com_results])
tv_avg_loss = np.mean([result[0] for result in tv_results])
tv_avg_acc = np.mean([result[1] for result in tv_results])

print(f"COM Average Loss: {com_avg_loss:.4f}, COM Average Accuracy: {com_avg_acc:.4f}")
print(f"TV Average Loss: {tv_avg_loss:.4f}, TV Average Accuracy: {tv_avg_acc:.4f}")
