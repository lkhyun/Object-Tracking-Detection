import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import torchsummary
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 커스텀 Dataset 클래스 정의 
class PVDFDataset2D(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32).to(device)  
        y = torch.tensor(self.labels[idx], dtype=torch.long).to(device)
        return x, y

def load_data(subject_files, group_indices=None):
    data_list = []
    labels_list = []

    for idx in (group_indices if group_indices is not None else range(len(subject_files))):
        file_path = subject_files[idx]
        with h5py.File(file_path, 'r') as f:
            for i in range(4):  # 각 셀을 순회 (4개의 클래스)
                data_ref = f['label_data'][i][0]
                num_samples = f[data_ref].shape[0]  # 데이터의 샘플 수 계산
                indices = np.arange(0, num_samples, 2)
                # 데이터 슬라이싱 및 전치
                data = np.array(f[data_ref][10:90, :, :]).T  # 데이터를 불러오고 transpose

                # Reshape 및 Transpose: (7500, 96, samples) -> (samples, 4, 7500, 24)
                num_samples = data.shape[2]  # 데이터의 샘플 수 계산
                reshaped_data = data.reshape((7500, 24, 4, num_samples))  # (7500, 24, 4, samples)
                reshaped_data = np.transpose(reshaped_data, (3, 2, 0, 1))  # (samples, 4, 7500, 24)
    
                # 레이블 할당
                label = 1 if i == 1 or i == 2 else 2 if i == 3 else 0

                data_list.append(reshaped_data)
                labels_list.append(np.full((reshaped_data.shape[0],), label))  # 클래스 레이블 추가

    data = np.concatenate(data_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    dataset = PVDFDataset2D(data, labels)
    return dataset

# 2D CNN 모델 정의
class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=(125, 3), stride=(25, 2), padding=(12, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(25, 2), stride=(5, 1), padding=(7, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 2), stride=(1, 1), padding=(2, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.6)

        self.fc = nn.Linear(32, 3)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 학습 함수
def train_model(model, train_loader_list, criterion, optimizer, scheduler, validation_loader, num_epochs=10):
    model.train()
    best_validation_loss = float('inf') 
    
    for epoch in range(num_epochs):
        total_correct = 0
        total_samples = 0
        running_loss = 0.0
        total_batches = 0 

        for train_loader in train_loader_list:
            for inputs, labels in train_loader:
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                total_batches += 1

        # 에폭 종료 후 평균 손실 계산
        average_loss = running_loss / total_batches
        train_accuracy = total_correct / total_samples * 100
        print(f'Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.2f}%, Loss: {average_loss:.4f}')

        validation_loss, validation_accuracy = validate_model(model, validation_loader, criterion)
        print(f'Epoch {epoch + 1}, Validation Accuracy: {validation_accuracy:.2f}%, Validation Loss: {validation_loss:.4f}')
        
        scheduler.step(validation_loss)
        print(f"Current Learning Rate: {scheduler.get_last_lr()}")

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), "2d_best_model.pth")  
            print(f"Best model saved at epoch {epoch + 1} with validation loss: {validation_loss:.4f}")


def validate_model(model, validation_loader, criterion):
    model.eval()
    total_correct = 0
    total_samples = 0
    running_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for inputs, labels in validation_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            total_batches += 1

    validation_loss = running_loss / total_batches
    validation_accuracy = total_correct / total_samples * 100
    return validation_loss, validation_accuracy

def test_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    test_accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("Confusion Matrix:\n", cm)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
    
    return test_accuracy

# 학습 및 평가 수행
def train_and_evaluate(subject_files, num_epochs=10, patience=5):
    random.shuffle(subject_files)

    test_subjects= subject_files[:6]  # 검증용
    validation_subjects  = subject_files[6:12]  # 테스트용
    train_subjects = subject_files[12:]  # 학습용

    validation_dataset = load_data(validation_subjects)
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)

    model = CNN2D().to(device)

    # class_weights = torch.tensor([1.0, 2.0, 1.5], device=device)  # 각 클래스에 대한 가중치 설정
    criterion = nn.CrossEntropyLoss()  # 클래스 가중치 적용

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

    train_groups = np.array_split(np.arange(len(train_subjects)), 7)
    train_loader_list = [DataLoader(load_data(train_subjects, group), batch_size=16, shuffle=True) for group in train_groups]

    train_model(model, train_loader_list, criterion, optimizer, scheduler, validation_loader, num_epochs)

    # 테스트 성능 평가
    print("\nTest Performance:")
    test_dataset = load_data(test_subjects)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    test_accuracy = test_model(model, test_loader)
    
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    print("\nModel Summary:")
    torchsummary.summary(CNN2D().to(device), (4, 7500, 24), device=str(device))

if __name__ == "__main__":
    set_seed(42)
    subject_files = [f'label_data_7500_96_{i}.mat' for i in range(1, 60) if i not in [38, 42, 57]]
    train_and_evaluate(subject_files, num_epochs=50, patience=50)