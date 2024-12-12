import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=64):
        super(VAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU()
            )
        
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        log_var = self.log_var_layer(encoded)
        z = self.reparameterize(mu, log_var)
        # 解码
        decoded = self.decoder(z)
        return decoded, mu, log_var, z

# 定义分类器
class VAEClassifier(nn.Module):
    def __init__(self, vae, latent_dim=64, num_classes=10):
        super(VAEClassifier, self).__init__()
        self.vae = vae
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        _, _, _, z = self.vae(x)
        logits = self.classifier(z)
        return logits, z

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))
])

train_dataset = datasets.MNIST(root="../Data/MNIST", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="../Data/MNIST", train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# 初始化模型、损失函数和优化器
input_dim = 784
hidden_dim = 256
latent_dim = 64
vae = VAE(input_dim, hidden_dim, latent_dim)
model = VAEClassifier(vae, latent_dim)

reconstruction_loss_fn = nn.BCELoss(reduction='sum')
classification_loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练过程
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 784)  # 展平
        target = target

        optimizer.zero_grad()
        
        # VAE前向传播
        recon_batch, mu, log_var, z = model.vae(data)
        recon_loss = reconstruction_loss_fn(recon_batch, data)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # 分类器前向传播
        logits, _ = model(data)
        class_loss = classification_loss_fn(logits, target)

        # 损失函数组合
        loss = recon_loss + kl_divergence + class_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset)}")

# 测试过程
def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 784)
            logits, _ = model(data)
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(f"Test Accuracy: {correct / len(test_loader.dataset) * 100:.2f}%")

# 主程序
for epoch in range(1, 21):
    train(epoch)
    test()
