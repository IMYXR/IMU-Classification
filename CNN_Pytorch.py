import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 数据加载
AC = np.load('./newdata/AC_array.npy')
AD = np.load('./newdata/AD_array.npy')
BC = np.load('./newdata/BC_array.npy')
BD = np.load('./newdata/BD_array.npy')
AC_eva=np.load('./evaluate_data/AC_eva.npy')
AD_eva=np.load('./evaluate_data/AD_eva.npy')
BC_eva=np.load('./evaluate_data/BC_eva.npy')
BD_eva=np.load('./evaluate_data/BD_eva.npy')
# 为每个段分配标签
AC_labels = [0] * len(AC)
AD_labels = [1] * len(AD)
BC_labels = [2] * len(BC)
BD_labels = [3] * len(BD)
AC_eva_labels = [0] * len(AC_eva)
AD_eva_labels = [1] * len(AD_eva)
BC_eva_labels = [2] * len(BC_eva)
BD_eva_labels = [3] * len(BD_eva)
# 合并数据和标签
data = np.vstack([AC, AD, BC, BD])
labels = AC_labels + AD_labels + BC_labels + BD_labels
eva_data = np.vstack([AC_eva, AD_eva, BC_eva, BD_eva])
eva_labels = AC_eva_labels + AD_eva_labels + BC_eva_labels + BD_eva_labels
#归一化
def min_max_normalize(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
data = min_max_normalize(data)
eva_data = min_max_normalize(eva_data)
# 转换为 PyTorch tensors
X = torch.tensor(data, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)

X_eva = torch.tensor(eva_data, dtype=torch.float32)
y_eva = torch.tensor(eva_labels, dtype=torch.long)
# 创建 DataLoader
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
valid_data = TensorDataset(X_eva, y_eva)
valid_loader = DataLoader(valid_data, batch_size=91, shuffle=True)


#加入注意力机制
class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.bias = bias
        self.attention = nn.Linear(feature_dim, 1, bias=bias)

    def forward(self, x):
        eij = self.attention(x)

        if self.bias:
            eij = torch.tanh(eij)

        a = torch.exp(eij)

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * a
        return torch.sum(weighted_input, 1)


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(18, 108, 5, padding='same', )
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool1d(2, 2, padding=1)
        self.conv2 = nn.Conv1d(108, 64, 5, padding='same')
        self.conv3 = nn.Conv1d(64, 32, 3, padding='same')
        self.attention = Attention(feature_dim=18, step_dim=5)
        self.fc1 = nn.Linear(2432, 256)  # 修改这一层的输出特征数
        self.relu1 = nn.LeakyReLU()  # 可以加入一个ReLU激活层
        self.fc2 = nn.Linear(256, 108)  # 新增一个全连接层
        self.dropout = nn.Dropout(0.5)  # Dropout 层，概率为0.5
        self.fc3 = nn.Linear(108, 4)  # 这是原来的第二个全连接层，现在变成第三个

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        # x = self.dropout(x)  # 应用 Dropout
        x = self.relu(self.conv2(x))
        # x = self.dropout(x)  # 应用 Dropout
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)  # 应用 Dropout
        x = self.relu1(self.fc1(x))
        # x = self.dropout(x)  # 应用 Dropout
        x = self.relu(self.fc2(x))
        x = self.dropout(x)  # 应用 Dropout
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#early stopping
history = 90
best_val_acc = 0.0
patience = 50  # 耐心值，即在提前停止之前等待的连续epoch数
epochs = 300

while best_val_acc < history:
    model = Net().to(device)  # 在循环内重新实例化模型
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    no_improve_epochs = 0  # 没有改善的epoch数
    best_val_acc = 0.0  # 重置最佳验证准确率

    for epoch in range(epochs):
        model.train()  # 开启训练模式
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # 训练过程
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        # 计算训练集的平均损失和准确率
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_predictions / total_predictions

        # 在验证集上评估模型
        model.eval()  # 开启评估模式
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%")

        # 检查是否有改善，并应用早停逻辑
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')  # 保存准确率最高的模型
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"No improvement for {patience} consecutive epochs, stopping training. Best Validation Accuracy: {best_val_acc}%")
                break

    if best_val_acc >= history:
        print(f"Reached accuracy higher than history. Best Validation Accuracy: {best_val_acc}%")
        break
