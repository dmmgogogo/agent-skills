# PyTorch Skill

## 📚 工具简介

**PyTorch** 是Facebook开发的开源深度学习框架,因其易用性和动态计算图而成为AI研究和生产的首选框架。

### 核心特性
- **动态计算图**: 灵活的模型构建
- **自动微分**: 简化梯度计算
- **GPU加速**: 无缝CUDA支持
- **丰富生态**: torchvision, torchaudio, torchtext等
- **生产部署**: TorchScript, ONNX支持
- **分布式训练**: 内置分布式训练支持

### GitHub信息
- **Stars**: 业界领先
- **社区**: 最活跃的深度学习社区
- **仓库**: https://github.com/pytorch/pytorch
- **官方文档**: https://pytorch.org/docs/

### 适用场景
✅ 深度学习研究
✅ 计算机视觉
✅ 自然语言处理(NLP)
✅ 强化学习
✅ GPT、Llama等大语言模型开发
✅ 生成式AI(GANs, Diffusion Models)

---

## 🔧 安装和配置

### CPU版本

```bash
# 使用pip安装
pip install torch torchvision torchaudio --break-system-packages
```

### GPU版本 (CUDA)

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --break-system-packages

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --break-system-packages
```

### 验证安装

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

---

## 💻 代码示例

### 1. 张量基础操作

```python
import torch

# 创建张量
x = torch.tensor([[1, 2], [3, 4]])
y = torch.zeros(2, 3)
z = torch.randn(2, 2)  # 随机正态分布

# 张量运算
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(a + b)  # 加法
print(a * b)  # 逐元素乘法
print(torch.dot(a, b))  # 点积

# 形状操作
x = torch.randn(2, 3, 4)
print(x.shape)
x_reshaped = x.view(2, 12)  # 重塑
x_transposed = x.permute(2, 0, 1)  # 转置

# GPU操作
if torch.cuda.is_available():
    x_gpu = x.cuda()  # 移到GPU
    x_cpu = x_gpu.cpu()  # 移回CPU
```

### 2. 自动微分

```python
# 启用梯度追踪
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1

# 反向传播
y.backward()
print(f"dy/dx = {x.grad}")  # 4 * 2 + 3 = 11

# 多变量
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.mean()

z.backward()
print(x.grad)
```

### 3. 构建神经网络

```python
import torch.nn as nn
import torch.nn.functional as F

# 定义网络
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 实例化模型
model = SimpleNet(input_size=784, hidden_size=128, num_classes=10)
print(model)

# 查看参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

### 4. 卷积神经网络(CNN)

```python
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = ConvNet()
```

### 5. 完整训练循环

```python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 准备数据
X_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型、损失函数、优化器
model = SimpleNet(784, 128, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_x, batch_y in dataloader:
        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
```

### 6. 模型保存和加载

```python
# 保存整个模型
torch.save(model, 'model.pth')
model = torch.load('model.pth')

# 只保存参数(推荐)
torch.save(model.state_dict(), 'model_weights.pth')
model = SimpleNet(784, 128, 10)
model.load_state_dict(torch.load('model_weights.pth'))

# 保存检查点
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载检查点
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### 7. 使用预训练模型

```python
import torchvision.models as models

# 加载预训练ResNet
resnet = models.resnet50(pretrained=True)

# 冻结参数
for param in resnet.parameters():
    param.requires_grad = False

# 修改最后一层用于迁移学习
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)  # 10个类别

# 只训练最后一层
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)
```

---

## 🎯 最佳实践

### 1. 设备管理

```python
# 自动选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将模型和数据移到设备
model = model.to(device)
inputs = inputs.to(device)
labels = labels.to(device)

# 或者使用上下文管理器
with torch.cuda.device(0):
    model = model.cuda()
```

### 2. 混合精度训练(节省显存)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()

    # 自动混合精度
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # 缩放损失并反向传播
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 3. 学习率调度

```python
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

# 阶梯式衰减
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# 余弦退火
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# 训练循环中使用
for epoch in range(num_epochs):
    train(...)
    scheduler.step()
```

### 4. 梯度裁剪(防止梯度爆炸)

```python
# 在optimizer.step()之前
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 5. 数据增强

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

---

## ⚠️ 常见问题和注意事项

### 问题1: CUDA Out of Memory

```python
# 解决方案:
# 1. 减小batch size
# 2. 使用混合精度训练
# 3. 梯度累积
accumulation_steps = 4
for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 4. 清理显存
torch.cuda.empty_cache()
```

### 问题2: 梯度消失/爆炸

```python
# 使用梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 使用Batch Normalization
self.bn = nn.BatchNorm2d(num_features)

# 使用残差连接
class ResidualBlock(nn.Module):
    def forward(self, x):
        return x + self.layer(x)
```

### 问题3: 模型不收敛

```python
# 检查清单:
# 1. 学习率是否合适
# 2. 数据是否标准化
# 3. 权重初始化是否正确
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

# 4. 添加梯度监控
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} gradient: {param.grad.norm()}")
```

---

## 📖 进阶资源

### 官方资源
- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [PyTorch文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch示例](https://github.com/pytorch/examples)

### 推荐课程
- fast.ai - Practical Deep Learning
- Stanford CS231n (使用PyTorch版本)
- PyTorch官方60分钟入门

### 相关库
- **torchvision**: 计算机视觉工具
- **torchaudio**: 音频处理
- **torchtext**: NLP工具
- **pytorch-lightning**: 高级训练框架

---

## 🔗 相关Skills

- **huggingface-skill**: Transformer模型
- **numpy-skill**: 数组操作基础
- **matplotlib-skill**: 训练可视化
- **jupyter-skill**: 交互式实验

---

**最后更新**: 2026-01-22
**版本**: 2.x
