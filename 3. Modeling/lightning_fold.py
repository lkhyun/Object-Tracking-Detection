import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import os
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, Subset
from sklearn.model_selection import GroupKFold

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
    def __init__(self, channel_num, paramArr, train_data, test_data):
        super(Conv2DLightning, self).__init__()
        self.paramArr = paramArr
        self.train_data = train_data
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
        return {'test_acc': correct / total * 100}  # Return test accuracy for averaging

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=4, num_workers=0, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=4, num_workers=0, shuffle=False)

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

# Create group labels for each person
groups_com = np.concatenate([[i] * 14 for i in range(1, 289)])
groups_tv = np.concatenate([[i] * 14 for i in range(1, 289)])

# Group K-Fold Cross Validation
gkf = GroupKFold(n_splits=5)

def run_group_kfold_cv(data, labels, groups, paramArr, max_epochs=10):
    fold = 1
    accuracies = []
    
    for train_index, test_index in gkf.split(data, labels, groups=groups):
        print(f"Fold {fold}")
        
        train_data = Subset(TensorDataset(data, labels), train_index)
        test_data = Subset(TensorDataset(data, labels), test_index)
        
        # Instantiate your model (code for model creation goes here)
        model = Conv2DLightning(channel_num=1, paramArr=paramArr, train_data=train_data, test_data=test_data)
        
        # Setup Trainer and train/validate/test the model
        trainer = pl.Trainer(max_epochs=max_epochs)
        trainer.fit(model)
        
        # Run test step and collect accuracy
        test_result = trainer.test(model)
        test_acc = test_result[0]['test_acc']  # Assuming 'test_acc' is the key in the returned dictionary
        accuracies.append(test_acc)
        
        fold += 1
    
    # Calculate and print average test accuracy across all folds
    avg_accuracy = np.mean(accuracies)
    print(f"\nAverage Test Accuracy: {avg_accuracy:.2f}%")

# Run GroupKFold CV for both datasets
print("Running Group K-Fold Cross Validation for COM Model")
run_group_kfold_cv(all_data_com, all_labels_com, groups_com, model_config_com)

print("Running Group K-Fold Cross Validation for TV Model")
run_group_kfold_cv(all_data_tv, all_labels_tv, groups_tv, model_config_tv)