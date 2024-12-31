import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

os.chdir("C:/Users/dlrkd/Desktop/hai-lab/model")

model_config_com = np.array([[32,64,128],     # channel each convolution layers
                         [2,2,2],       # kernel size of convolution layer
                         [1,1,1],       # stride size of convolution layer
                         [2,2,2],       # kernel size of pooling layer
                         [1,1,1],       # stride size of pooling layer
                         [128,64,32],    # channel of denseblock layer
                         [9,9,9]        # class
                         ])
model_config_tv = np.array([[32,64,128],     # channel each convolution layers
                         [2,2,2],       # kernel size of convolution layer
                         [1,1,1],       # stride size of convolution layer
                         [2,2,2],       # kernel size of pooling layer
                         [1,1,1],       # stride size of pooling layer
                         [128,64,32],    # channel of denseblock layer
                         [7,7,7]        # class
                         ])

class conv2D_model(nn.Module):
    def __init__(self, channel_num, paramArr):
        super(conv2D_model, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=channel_num, out_channels=paramArr[0][0], kernel_size=(2, paramArr[1][0]), stride=(1, paramArr[2][0]), padding='valid', bias=False),
            nn.BatchNorm2d(paramArr[0][0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, paramArr[3][0]), stride=(1, paramArr[4][0]))
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=paramArr[0][0], out_channels=paramArr[0][1], kernel_size=(2, paramArr[1][1]), stride=(1, paramArr[2][1]), padding='valid', bias=False),
            nn.BatchNorm2d(paramArr[0][1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, paramArr[3][1]), stride=(1, paramArr[4][1]))
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=paramArr[0][1], out_channels=paramArr[0][2], kernel_size=(2, paramArr[1][2]), stride=(1, paramArr[2][2]), padding='valid', bias=False),
            nn.BatchNorm2d(paramArr[0][2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, paramArr[3][2]), stride=(1, paramArr[4][2]))
        )

        self.flatten = nn.Flatten()

        # Calculate the size of the input for the first dense layer
        dummy_input = torch.zeros(1, channel_num, 51, 72)  # Create a dummy input tensor with the same shape as the expected input
        dummy_output = self.convblock3(self.convblock2(self.convblock1(dummy_input)))  # Pass the dummy input through the convolutional blocks
        n_features = dummy_output.numel()  # Calculate the total number of features in the output tensor

        self.denseblock = nn.Sequential(
            nn.Linear(n_features, paramArr[5][0]),  # Use the calculated number of features as the input size for the first dense layer
            nn.ReLU(),
            nn.Linear(paramArr[5][0], paramArr[5][1]),
            nn.ReLU(),
            nn.Linear(paramArr[5][1], paramArr[5][2]),
            nn.ReLU()
        )
        self.out = nn.Linear(paramArr[5][2], paramArr[6][0])

        # Apply weight initialization
        self._initialize_weights()

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.flatten(x)
        x = self.denseblock(x)
        x = self.out(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Load and preprocess data
def load_data(file_path):
    full_data = np.load(file_path)
    data = full_data[:, :-1].reshape(-1, 1, 51, 72)  # Reshape
    #print(data.shape)
    labels = full_data[:, -1]
    #print(labels.shape)
    
    # Normalize data(z-score)
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std == 0] = 1
    data = (data - mean) / std
    
    return data, labels

# Define the models for each dataset (com and tv)
model_com = conv2D_model(1,model_config_com)
model_tv = conv2D_model(1,model_config_tv)

criterion_com = nn.CrossEntropyLoss()
criterion_tv = nn.CrossEntropyLoss()
optimizer_com = torch.optim.Adam(model_com.parameters())
optimizer_tv = torch.optim.Adam(model_tv.parameters())

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

# Setup datasets and dataloaders
train_dataset_com = TensorDataset(all_data_com[:3193], all_labels_com[:3193]) #3192장 훈련 38명
valid_dataset_com = TensorDataset(all_data_com[3193:3613], all_labels_com[3193:3613]) #420장 검증 5명
test_dataset_com = TensorDataset(all_data_com[3613:4033], all_labels_com[3613:4033]) #420장 테스트 5명
train_loader_com = DataLoader(train_dataset_com, batch_size=8, shuffle=False)
valid_loader_com = DataLoader(valid_dataset_com, batch_size=8, shuffle=False)
test_loader_com = DataLoader(test_dataset_com, batch_size=1, shuffle=False)

train_dataset_tv = TensorDataset(all_data_tv[:3193], all_labels_tv[:3193])
valid_dataset_tv = TensorDataset(all_data_tv[3193:3613], all_labels_tv[3193:3613])
test_dataset_tv = TensorDataset(all_data_tv[3613:4033], all_labels_tv[3613:4033])
train_loader_tv = DataLoader(train_dataset_tv, batch_size=8, shuffle=False)
valid_loader_tv = DataLoader(valid_dataset_tv, batch_size=8, shuffle=False)
test_loader_tv = DataLoader(test_dataset_tv, batch_size=1, shuffle=False)


def train_and_evaluate(model, train_loader, valid_loader, test_loader, optimizer, criterion):
    num_epochs = 10
    train_losses, valid_losses = [], []
    valid_accuracies = []
    test_losses = []
    test_accuracies = []
    final_test_predictions = []
    final_test_targets = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        correct, total = 0, 0
        valid_loss = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
            avg_valid_loss = valid_loss / len(valid_loader)
            valid_losses.append(avg_valid_loss)
            valid_accuracy = 100 * correct / total
            valid_accuracies.append(valid_accuracy)

        if epoch == num_epochs - 1:
            correct, total = 0, 0
            test_loss = 0
            test_predictions = []
            test_targets = []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                    test_predictions.extend(predicted.tolist())
                    test_targets.extend(targets.tolist())
                    
            avg_test_loss = test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            test_accuracy = 100 * correct / total
            test_accuracies.append(test_accuracy)
            final_test_predictions = test_predictions
            final_test_targets = test_targets

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%")

    print(f"Final Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Plotting the losses and accuracies
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(valid_accuracies, label='Valid Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation and Test Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Generating confusion matrix for final test predictions
    cm = confusion_matrix(final_test_targets, final_test_predictions)
    plt.figure(figsize=(8, 6))
    classes = np.unique(final_test_targets)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# 예시: Training and evaluating model_com
print("Training and evaluating model_com:")
train_and_evaluate(model_com, train_loader_com, valid_loader_com, test_loader_com, optimizer_com, criterion_com)

# 예시: Training and evaluating model_tv
print("Training and evaluating model_tv:")
train_and_evaluate(model_tv, train_loader_tv, valid_loader_tv, test_loader_tv, optimizer_tv, criterion_tv)
