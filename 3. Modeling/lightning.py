import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import os
import pytorch_lightning as pl
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

class Conv2DLightning(pl.LightningModule):
    def __init__(self, channel_num, paramArr, train_data, valid_data, test_data):
        super(Conv2DLightning, self).__init__()
        self.paramArr = paramArr
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=channel_num, out_channels=self.paramArr[0][0], kernel_size=(2, self.paramArr[1][0]), stride=(1, self.paramArr[2][0]), padding='valid', bias=False),
            nn.BatchNorm2d(self.paramArr[0][0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, self.paramArr[3][0]), stride=(1, self.paramArr[4][0]))
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=self.paramArr[0][0], out_channels=self.paramArr[0][1], kernel_size=(2, self.paramArr[1][1]), stride=(1, self.paramArr[2][1]), padding='valid', bias=False),
            nn.BatchNorm2d(self.paramArr[0][1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, self.paramArr[3][1]), stride=(1, self.paramArr[4][1]))
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=self.paramArr[0][1], out_channels=self.paramArr[0][2], kernel_size=(2, self.paramArr[1][2]), stride=(1, self.paramArr[2][2]), padding='valid', bias=False),
            nn.BatchNorm2d(self.paramArr[0][2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, self.paramArr[3][2]), stride=(1, self.paramArr[4][2]))
        )

        self.flatten = nn.Flatten()

        # Calculate the size of the input for the first dense layer
        dummy_input = torch.zeros(1, channel_num, 51, 72)  # Create a dummy input tensor with the same shape as the expected input
        dummy_output = self.convblock3(self.convblock2(self.convblock1(dummy_input)))  # Pass the dummy input through the convolutional blocks
        n_features = dummy_output.numel()  # Calculate the total number of features in the output tensor

        self.denseblock = nn.Sequential(
            nn.Linear(n_features, self.paramArr[5][0]),  # Use the calculated number of features as the input size for the first dense layer
            nn.ReLU(),
            nn.Linear(self.paramArr[5][0], self.paramArr[5][1]),
            nn.ReLU(),
            nn.Linear(self.paramArr[5][1], self.paramArr[5][2]),
            nn.ReLU()
        )
        self.out = nn.Linear(self.paramArr[5][2], self.paramArr[6][0])

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

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        self.log('val_loss', loss, prog_bar=True)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        self.log('val_acc', correct / total * 100, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        self.log('test_loss', loss)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        self.log('test_acc', correct / total * 100)

        # Return predicted and true labels for confusion matrix computation
        return {'predicted': predicted, 'targets': targets}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=8, num_workers=0, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=8, num_workers=0, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, num_workers=0, shuffle=False)

# Load and preprocess data
def load_data(file_path):
    full_data = np.load(file_path)
    data = full_data[:, :-1].reshape(-1, 1, 51, 72)  # Reshape
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

# Setup datasets
train_dataset_com = TensorDataset(all_data_com[:3193], all_labels_com[:3193])
valid_dataset_com = TensorDataset(all_data_com[3193:3613], all_labels_com[3193:3613])
test_dataset_com = TensorDataset(all_data_com[3613:4033], all_labels_com[3613:4033])

train_dataset_tv = TensorDataset(all_data_tv[:3193], all_labels_tv[:3193])
valid_dataset_tv = TensorDataset(all_data_tv[3193:3613], all_labels_tv[3193:3613])
test_dataset_tv = TensorDataset(all_data_tv[3613:4033], all_labels_tv[3613:4033])

# Instantiate Lightning modules
model_com_lightning = Conv2DLightning(channel_num=1, paramArr=model_config_com, train_data=train_dataset_com, valid_data=valid_dataset_com, test_data=test_dataset_com)
model_tv_lightning = Conv2DLightning(channel_num=1, paramArr=model_config_tv, train_data=train_dataset_tv, valid_data=valid_dataset_tv, test_data=test_dataset_tv)

# Setup Trainer
trainer_com = pl.Trainer(max_epochs=10)
trainer_tv = pl.Trainer(max_epochs=10)

# Train model_com_lightning
print("Training model_com_lightning:")
trainer_com.fit(model_com_lightning)

# Validate model_com_lightning
print("Validating model_com_lightning:")
trainer_com.validate(model_com_lightning)

print("Testing model_com_lightning:")
trainer_com.test(model_com_lightning)

# Train model_tv_lightning
print("Training model_tv_lightning:")
trainer_tv.fit(model_tv_lightning)

# Validate model_tv_lightning
print("Validating model_tv_lightning:")
trainer_tv.validate(model_tv_lightning)

print("Testing model_tv_lightning:")
trainer_tv.test(model_tv_lightning)

# Function to compute confusion matrix
def compute_confusion_matrix(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.tolist())
            all_targets.extend(targets.tolist())

    cm = confusion_matrix(all_targets, all_preds)
    return cm

# Compute confusion matrices for both models
cm_com = compute_confusion_matrix(model_com_lightning, model_com_lightning.test_dataloader())
cm_tv = compute_confusion_matrix(model_tv_lightning, model_tv_lightning.test_dataloader())

# Print confusion matrices to check their content
print("Confusion Matrix - model_com_lightning:")
print(cm_com)

print("Confusion Matrix - model_tv_lightning:")
print(cm_tv)

# Replace NaN values with zeros (if any)
cm_com = np.nan_to_num(cm_com)
cm_tv = np.nan_to_num(cm_tv)

# Plot confusion matrix for model_com_lightning
plt.figure(figsize=(8, 6))
sns.heatmap(cm_com, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - model_com_lightning')
plt.show()

# Plot confusion matrix for model_tv_lightning
plt.figure(figsize=(8, 6))
sns.heatmap(cm_tv, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - model_tv_lightning')
plt.show()