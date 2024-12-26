import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
import torchsummary

# GPU 할당 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 커스텀 Dataset 클래스 정의 (2D CNN용 데이터 로드 및 변환)
class PVDFDataset2D(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.labels = []
        
        # HDF5 파일 열기
        with h5py.File(file_path, 'r') as f:
            for i in range(4):  # 각 셀을 순회 (4개의 클래스)
                data_ref = f['label_data'][i][0]
                data = np.array(f[data_ref]).T  # 데이터를 불러오고 transpose

                # Reshape 및 Transpose: (7500, 96, samples) -> (samples, 4, 7500, 24)
                num_samples = data.shape[2]
                reshaped_data = data.reshape((7500, 24, 4, num_samples))  # (7500, 24, 4, samples)
                reshaped_data = np.transpose(reshaped_data, (3, 2, 0, 1))  # (samples, 4, 7500, 24)
                
                # 레이블 할당
                if i == 1 or i == 2:
                    label = 1
                elif i == 3:
                    label = 2
                else:
                    label = 0
                
                self.data.append(reshaped_data)
                self.labels.append(np.full((reshaped_data.shape[0],), label))  # 클래스 레이블 추가
        
        # 리스트를 배열로 변환하여 하나의 데이터셋으로 통합
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)  # (samples, 4, 7500, 24) 형태로 변환
        y = torch.tensor(self.labels[idx], dtype=torch.long)  # 클래스 레이블을 long으로 변환
        return x, y

# 2D CNN 모델 정의 (모델 개선: BatchNorm, Dropout, Global Avg Pooling 추가)
class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.dropout = nn.Dropout(0.5)  # Dropout to prevent overfitting
        
        self.fc1 = nn.Linear(128, 128)  # Output of global avg pool is 128 channels
        self.fc2 = nn.Linear(128, 3)  # 3-class classification

    def forward(self, x):
        x = self.pool1(self.bn1(torch.relu(self.conv1(x))))
        x = self.pool2(self.bn2(torch.relu(self.conv2(x))))
        x = self.pool3(self.bn3(torch.relu(self.conv3(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 모델과 데이터를 GPU로 이동시키는 함수
def to_device(data, device):
    if isinstance(data, (tuple, list)):
        return [to_device(d, device) for d in data]
    return data.to(device, non_blocking=True)

# 데이터를 로드하는 함수 (그룹별 데이터 로드)
def load_group_data(subject_files, group_indices):
    datasets = []
    for index in group_indices:
        file_path = subject_files[index]
        dataset = PVDFDataset2D(file_path)  # 각 파일을 Dataset으로 변환
        datasets.append(dataset)
    return torch.utils.data.ConcatDataset(datasets)

# 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.train()
    total_correct = 0
    total_samples = 0
    model.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = to_device((inputs, labels), device)  # 데이터를 GPU로 이동
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 손실 계산
            running_loss += loss.item()

            # Train accuracy 계산
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        total_correct += correct
        total_samples += total
        train_accuracy = total_correct / total_samples * 100
        print(f'Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.2f}%,  Loss: {running_loss / len(train_loader):.4f}')

# 성능 지표 추가: Confusion Matrix, Precision, Recall, F1-Score 및 테스트
def test_model(model, test_loader, device='cpu'):
    model.eval()
    all_labels = []
    all_preds = []
    model.to(device)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = to_device((inputs, labels), device)  # 데이터를 GPU로 이동
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Confusion Matrix 및 성능 지표 출력
    cm = confusion_matrix(all_labels, all_preds)
    test_accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("Confusion Matrix:\n", cm)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
    
    return test_accuracy, precision, recall, f1

# KFold + Group 학습 및 최종 평가 (평균 ± 표준편차 계산)
def train_model_by_kfold_with_groups(model, subject_files, criterion, optimizer, num_epochs=10, device='cpu'):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for fold, (train_index, test_index) in enumerate(kf.split(subject_files)):
        print(f'Fold {fold + 1}')
        
        # Train 데이터를 그룹으로 나눔
        train_groups = np.array_split(train_index, 5)
        test_group = test_index

        # 모델 초기화
        model = CNN2D().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 그룹 단위로 학습
        for group_idx, group in enumerate(train_groups):
            print(f"Training Group {group_idx + 1}")
            
            train_dataset = load_group_data(subject_files, group)
            test_dataset = load_group_data(subject_files, test_group)

            # DataLoader로 변환
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

            # 모델 학습
            train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs, device=device)
        
        # 테스트 성능 평가
        accuracy, precision, recall, f1 = test_model(model, test_loader, device)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # 최종 결과 계산 (평균 ± 표준편차)
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)
    
    print(f"Final Result: Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
    print(f"Final Precision: {mean_precision:.2f}")
    print(f"Final Recall: {mean_recall:.2f}")
    print(f"Final F1-Score: {mean_f1:.2f}")

# 모델 및 학습 관련 설정
if __name__ == "__main__":
    subject_files = [f'label_data_7500_96_{i}.mat' for i in range(1, 60) if i not in [38, 42, 57]]

    model = CNN2D().to(device)  # 모델 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # KFold + Group 학습 및 최종 평가
    train_model_by_kfold_with_groups(model, subject_files, criterion, optimizer, num_epochs=5, device=device)

    # 모델 summary 출력
    torchsummary.summary(model, (4, 7500, 24), device=str(device))
