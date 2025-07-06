import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed
torch.manual_seed(200)
np.random.seed(200)

# Generate the 8-mode Gaussian mixture dataset
def generate_gaussian_mixture(n_samples=50000, variance=25e-4):
    centers = np.array([
        [np.cos(theta), np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, 9)[:-1]
    ])  # 8 centers on a unit circle

    data = []
    for _ in range(n_samples):
        center = centers[np.random.choice(len(centers))]
        sample = np.random.normal(loc=center, scale=np.sqrt(variance), size=(2,))
        data.append(sample)

    data = np.array(data)
    return data

# Visualize dataset
def plot_dataset(data, filename="dataset.png"):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], alpha=0.3)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title("8-Mode Gaussian Mixture Dataset")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

# Define Generator
def get_generator():
    return nn.Sequential(
        nn.Linear(2, 128),
        nn.LeakyReLU(0.2),
        nn.Linear(128, 128),
        nn.LeakyReLU(0.2),
        nn.Linear(128, 2)
    ).to(device)

# Define Discriminator
def get_discriminator():
    return nn.Sequential(
        nn.Linear(2, 128),
        nn.LeakyReLU(0.2),
        nn.Linear(128, 128),
        nn.LeakyReLU(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid()
    ).to(device)

# Training loop for GAN
def train_gan(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion, 
              n_epochs=400, saturating=True, log_file="training_log.txt", sample_intervals=[50, 100, 200, 300, 400]):
    mode_coverage = []
    generated_samples = {}
    
    with open(log_file, "w") as log:
        for epoch in range(n_epochs):
            for real_samples in dataloader:
                real_samples = real_samples.to(device)
                
                # Train Discriminator
                optimizer_D.zero_grad()
                real_preds = discriminator(real_samples)
                real_loss = criterion(real_preds, torch.ones_like(real_preds))
                
                noise = torch.randn(real_samples.size(0), 2).to(device)
                fake_samples = generator(noise)
                fake_preds = discriminator(fake_samples.detach())
                fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))
                
                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_D.step()
                
                # Train Generator
                optimizer_G.zero_grad()
                fake_preds = discriminator(fake_samples)
                
                if saturating:
                    g_loss = criterion(fake_preds, torch.ones_like(fake_preds))
                else:
                    g_loss = -torch.log(fake_preds).mean()
                
                g_loss.backward()
                optimizer_G.step()
            
            # Compute mode coverage every epoch
            with torch.no_grad():
                gen_samples = generator(torch.randn(10000, 2).to(device)).cpu().numpy()
                mode_count = count_modes(gen_samples)
                mode_coverage.append(mode_count)
                
                if epoch+1 in sample_intervals:
                    generated_samples[epoch+1] = gen_samples
            
            log.write(f"Epoch {epoch+1}/{n_epochs}, Mode Coverage: {mode_count}\n")
            print(f"Epoch {epoch+1}/{n_epochs}, Mode Coverage: {mode_count}")
    
    return mode_coverage, generated_samples

# Count mode coverage
def count_modes(samples, threshold=0.1):
    centers = np.array([
        [np.cos(theta), np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, 9)[:-1]
    ])
    covered_modes = 0
    for center in centers:
        distances = np.linalg.norm(samples - center, axis=1)
        if np.sum(distances < threshold) > 50:
            covered_modes += 1
    return covered_modes

# Function to visualize mode coverage
def plot_mode_coverage(mode_coverage, filename="mode_coverage.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(mode_coverage, label="Mode Coverage")
    plt.xlabel("Epoch")
    plt.ylabel("Modes Covered")
    plt.legend()
    plt.title("Mode Coverage Over Training")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

# Function to visualize discriminator heatmap
def plot_discriminator_heatmap(discriminator, epoch, filename):
    grid_x, grid_y = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-1.5, 1.5, 100))
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        preds = discriminator(grid_tensor).cpu().numpy().reshape(100, 100)
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(preds, cmap="coolwarm", xticklabels=False, yticklabels=False)
    plt.title(f"Discriminator Heatmap - Epoch {epoch}")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

# Function to overlay generated samples on mode coverage plot
def plot_mode_coverage_with_samples(mode_coverage, generated_samples, filename="saturating_mode_coverage.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(mode_coverage, label="Mode Coverage")
    plt.xlabel("Epoch")
    plt.ylabel("Modes Covered")
    plt.legend()
    plt.title("Mode Coverage Over Training")
    
    for epoch, samples in generated_samples.items():
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, label=f"Epoch {epoch}")
    
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

# Function to compute success and failure rates
def compute_success_failure_rates(generator, discriminator, dataloader, num_seeds=50):
    success_count = 0
    failure_count = 0
    
    for _ in range(num_seeds):
        generator.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        discriminator.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        
        optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
        optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)
        criterion = nn.BCELoss()
        
        mode_coverage, _ = train_gan(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion, n_epochs=400)
        
        if mode_coverage[-1] == 8:
            success_count += 1
        if mode_coverage[-1] == 0:
            failure_count += 1
    
    success_rate = (success_count / num_seeds) * 100
    failure_rate = (failure_count / num_seeds) * 100
    
    return success_rate, failure_rate

# Function to generate samples from trained models for different seeds
def generate_seed_samples(generator, real_data, num_seeds=6, filename="seed_samples.png"):
    plt.figure(figsize=(12, 6))
    
    for i in range(num_seeds):
        generator.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        noise = torch.randn(1000, 2).to(device)
        samples = generator(noise).cpu().detach().numpy()
        
        plt.subplot(2, num_seeds+1, i+1)
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3)
        plt.title(f"Seed {i+1}")
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
    
    # Add target real samples as the last plot
    plt.subplot(2, num_seeds+1, num_seeds+1)
    plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.3, color='red')
    plt.title("Target (Real Data)")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

# Main execution
if __name__ == "__main__":
    data = generate_gaussian_mixture()
    plot_dataset(data)
    
    dataset = torch.tensor(data, dtype=torch.float32)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    generator = get_generator()
    discriminator = get_discriminator()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    # Train Saturating GAN
    print("Training Saturating GAN...")
    mode_coverage_sat, generated_samples_sat = train_gan(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion, saturating=True, log_file="sat_gan_log.txt")
    plot_mode_coverage_with_samples(mode_coverage_sat, generated_samples_sat, filename="saturating_mode_coverage.png")

    # Generate heatmaps at different epochs
    for epoch in [50, 100, 200, 300, 400]:
        plot_discriminator_heatmap(discriminator, epoch, filename=f"saturating_discriminator_heatmap_epoch{epoch}.png")
    
    # Train Non-Saturating GAN
    generator = get_generator()
    discriminator = get_discriminator()
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)
    
    print("Training Non-Saturating GAN...")
    mode_coverage_ns, _ = train_gan(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion, saturating=False, log_file="ns_gan_log.txt")
    
    # Compute success and failure rates
    success_rate, failure_rate = compute_success_failure_rates(generator, discriminator, dataloader)
    print(f"Success Rate: {success_rate}%")
    print(f"Failure Rate: {failure_rate}%")
    
    # Generate samples for different seeds
    generate_seed_samples(generator, data, filename="seed_samples.png")