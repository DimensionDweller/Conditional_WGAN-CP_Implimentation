import torch
from dataset import CustomDataset
from model import Generator, Discriminator
from train import train
from utils import weights_init, compute_gradient_penalty

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    BATCH_SIZE = 128
    Z_DIM = 100
    FEATURES_G=64
    NUM_CLASSES=3
    EPOCHS = 50
    CRITIC_ITERATIONS = 5
    LAMBDA_GP = 10
    SAMPLE_SIZE = 8
    CHECKPOINT_PATH = None  # Specify if you want to continue training from a checkpoint

    # Load the dataset
    data_dir = "Animal Faces/Train"
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CustomDataset(root_dir=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize the models
    generator = Generator(z_dim=Z_DIM, channels_img=3, features_g=FEATURES_G).to(device)
    discriminator = Discriminator(channels_img=3, features_d=64).to(device)

    # Initialize the optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Train the models
    train(generator, discriminator, optimizer_G, optimizer_D, dataloader, device, EPOCHS, CRITIC_ITERATIONS, LAMBDA_GP, NUM_CLASSES, Z_DIM, SAMPLE_SIZE, CHECKPOINT_PATH)

if __name__ == "__main__":
    main()
