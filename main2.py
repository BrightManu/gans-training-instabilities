import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed
torch.manual_seed(200)
np.random.seed(200)

# Generate the 8-mode 2D Gaussian Mixture Ring dataset
def Generate_Gaussian_ring(total_samples=75000, variance=1e-4):
    centers = np.array([
        [np.cos(2*np.pi*i/8), np.sin(2*np.pi*i/8)] for i in range(8)
    ])
    
    data = []
    for _ in range(total_samples):
        center = centers[np.random.choice(len(centers))]
        sample = np.random.normal(loc=center, scale=np.sqrt(variance), size=(2,))
        data.append(sample)
    return np.array(data)

# Visualize dataset
def plot_dataset(data, filename="Prob_3_dataset.png"):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], c="green", alpha=0.3)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title("8-Mode Gaussian Mixture Dataset")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

# Define the Generator and Discriminator (Vanilla GAN)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 400),
            nn.LeakyReLU(0.2),
            nn.Linear(400, 400),
            nn.LeakyReLU(0.2),
            nn.Linear(400, 400),
            nn.LeakyReLU(0.2),
            nn.Linear(400, 2)
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Define the Critic (WGAN)
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Training functions with loss tracking
def train_gan(generator, discriminator, dataloader, num_epochs=300, lr=1e-4):
    generator, discriminator = generator.to(device), discriminator.to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    
    d_losses, g_losses = [], []
    
    for epoch in range(num_epochs):
        epoch_d_loss, epoch_g_loss = 0, 0
        for real_samples, in dataloader:
            real_samples = real_samples.to(device)
            batch_size = real_samples.shape[0]
            noise = torch.randn(batch_size, 2).to(device)
            fake_samples = generator(noise)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = loss_fn(discriminator(real_samples), torch.ones(batch_size, 1).to(device))
            fake_loss = loss_fn(discriminator(fake_samples.detach()), torch.zeros(batch_size, 1).to(device))
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()
            epoch_d_loss += d_loss.item()
            
            # Train Generator
            optimizer_G.zero_grad()
            g_loss = loss_fn(discriminator(fake_samples), torch.ones(batch_size, 1).to(device))
            g_loss.backward()
            optimizer_G.step()
            epoch_g_loss += g_loss.item()

        d_losses.append(epoch_d_loss / len(dataloader))
        g_losses.append(epoch_g_loss / len(dataloader))
        print(f"Epoch {epoch + 1}/{num_epochs}: Vanilla GAN Loss -> D: {d_losses[-1]:.4f}, G: {g_losses[-1]:.4f}")
    
    return d_losses, g_losses

# Critic training for WGAN
def train_wgan(generator, critic, dataloader, num_epochs=300, lr=1e-4, clip_value=0.01, n_critic=5, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, critic = generator.to(device), critic.to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_C = optim.Adam(critic.parameters(), lr=lr)
    
    c_losses, g_losses = [], []
    
    for epoch in range(num_epochs):
        epoch_c_loss, epoch_g_loss = 0, 0
        for i, (real_samples,) in enumerate(dataloader):
            real_samples = real_samples.to(device)
            batch_size = real_samples.shape[0]
            noise = torch.randn(batch_size, 2).to(device)
            fake_samples = generator(noise)
            
            # Train Critic
            optimizer_C.zero_grad()
            c_loss = -(torch.mean(critic(real_samples)) - torch.mean(critic(fake_samples.detach())))
            c_loss.backward()
            optimizer_C.step()
            epoch_c_loss += c_loss.item()
            
            # Clip weights of the critic
            for p in critic.parameters():
                p.data.clamp_(-clip_value, clip_value)
            
            # Train Generator every n_critic steps
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                g_loss = -torch.mean(critic(generator(noise)))
                g_loss.backward()
                optimizer_G.step()
                epoch_g_loss += g_loss.item()

        c_losses.append(epoch_c_loss / len(dataloader))
        g_losses.append(epoch_g_loss / (len(dataloader) / n_critic))
        print(f"Epoch {epoch + 1}/{num_epochs}: WGAN Loss -> Critic: {c_losses[-1]:.4f}, Generator: {g_losses[-1]:.4f}")
    
    return c_losses, g_losses

# Visualization functions
def plot_combined_losses(gan_losses, wgan_losses, filename="loss_plot.png"):
    plt.figure(figsize=(6,4))
    plt.plot(gan_losses, label="Vanilla GAN Loss", linestyle='dashed')
    plt.plot(wgan_losses, label="WGAN Loss", linestyle='solid')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Losses for Vanilla GAN & WGAN")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_progress(generator, epochs, title, filename_prefix="progress_plot"):
    fig, axes = plt.subplots(len(epochs), 2, figsize=(8, 4 * len(epochs)))

    for i, epoch in enumerate(epochs):
        with torch.no_grad():
            noise = torch.randn(5000, 2)
            samples = generator(noise).cpu().numpy()

        sns.kdeplot(x=samples[:, 0], y=samples[:, 1], fill=True, cmap="Blues", ax=axes[i, 0])
        axes[i, 0].set_title(f"{title} KDE - Epoch {epoch}")

        axes[i, 1].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=2, color='blue')
        axes[i, 1].set_title(f"{title} Samples - Epoch {epoch}")

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_{title}.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_samples(generator, title, filename="samples_plot.png"):
    with torch.no_grad():
        noise = torch.randn(5000, 2)
        samples = generator(noise).cpu().numpy()
    plt.figure(figsize=(6,6))
    sns.kdeplot(x=samples[:, 0], y=samples[:, 1], fill=True, cmap="Blues")
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=2)
    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def compute_mode_coverage(generator):
    with torch.no_grad():
        noise = torch.randn(5000, 2)
        samples = generator(noise).cpu().numpy()
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=8, random_state=42).fit(samples)
    unique_clusters = len(np.unique(kmeans.labels_))
    return unique_clusters

# Main function
if __name__ == "__main__":
    data = Generate_Gaussian_ring()
    dataloader = DataLoader(TensorDataset(torch.tensor(data, dtype=torch.float32)), batch_size=64, shuffle=True)

    plot_dataset(data, filename="Prob_3_dataset.png")

    print("Training Vanilla GAN...")
    gan_generator = Generator()
    gan_discriminator = Discriminator()
    gan_losses, _ = train_gan(gan_generator, gan_discriminator, dataloader)

    print("Training WGAN...")
    wgan_generator = Generator()
    wgan_critic = Critic()
    wgan_losses, _ = train_wgan(wgan_generator, wgan_critic, dataloader)
    
    plot_combined_losses(gan_losses, wgan_losses)
    
    plot_samples(gan_generator, "Vanilla GAN Generated Samples")
    plot_samples(wgan_generator, "WGAN Generated Samples")
    
    epochs = [0, 1, 5, 10, 20, 50, 100]
    plot_progress(gan_generator, epochs, "Vanilla GAN")
    plot_progress(wgan_generator, epochs, "WGAN")
    
    gan_mode_coverage = compute_mode_coverage(gan_generator)
    wgan_mode_coverage = compute_mode_coverage(wgan_generator)
    print(f"Vanilla GAN Mode Coverage: {gan_mode_coverage}/8")
    print(f"WGAN Mode Coverage: {wgan_mode_coverage}/8")
