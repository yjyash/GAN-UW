from src.dataloader import get_dataloader
from src.models.generator import Generator
from src.models.discriminator import Discriminator
import torch
import torch.nn as nn
import yaml
import os
from torch.optim import Adam
from src.dataloader import get_dataloader
from utils import log_losses, save_checkpoint, save_generated_images, calculate_metrics, display_metrics

def save_checkpoint(generator, discriminator, epoch, optimizer_g, optimizer_d, checkpoint_dir="../checkpoints/"):
    # Create directory if it doesn't exist -> done successfully
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save the model weights and optimizer states ->done successfully
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
    }, os.path.join(checkpoint_dir, f"epoch_{epoch}.pth"))

def load_checkpoint(checkpoint_path, generator, discriminator, optimizer_g, optimizer_d, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load generator state
    generator_state_dict = generator.state_dict()
    pretrained_generator_dict = checkpoint['generator_state_dict']

    # Filter out mismatched layers and load compatible ones
    matched_generator_dict = {
        k: v for k, v in pretrained_generator_dict.items()
        if k in generator_state_dict and generator_state_dict[k].size() == v.size()
    }
    generator_state_dict.update(matched_generator_dict)
    generator.load_state_dict(generator_state_dict)
    print(f"Generator: Loaded {len(matched_generator_dict)}/{len(pretrained_generator_dict)} layers.")

    # Load discriminator state
    discriminator_state_dict = discriminator.state_dict()
    pretrained_discriminator_dict = checkpoint['discriminator_state_dict']

    # Filter out mismatched layers for discriminator
    matched_discriminator_dict = {
        k: v for k, v in pretrained_discriminator_dict.items()
        if k in discriminator_state_dict and discriminator_state_dict[k].size() == v.size()
    }
    discriminator_state_dict.update(matched_discriminator_dict)
    discriminator.load_state_dict(discriminator_state_dict)
    print(f"Discriminator: Loaded {len(matched_discriminator_dict)}/{len(pretrained_discriminator_dict)} layers.")

    # Load optimizer states
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

    start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
    print(f"Checkpoint loaded: Resuming from epoch {start_epoch}")
    return start_epoch


def train():
        
    # Loading the configuration file -> Done successfully
        config_path = "../configs/config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # print(config)

    # GPU Readiness line -> Done successfully
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

#     # Making GAn READY -> done successfully
        generator = Generator().to(device)
        discriminator = Discriminator().to(device)

#     # Load dataloader -> ?
        dataloader = get_dataloader(config_path)

#     # Optimizers and loss setting -> done successfully
        optimizer_g = Adam(generator.parameters(), lr=config['train']['lr'], betas=(0.5, 0.999))
        optimizer_d = Adam(discriminator.parameters(), lr=config['train']['lr'], betas=(0.5, 0.999))
        loss_fn = nn.MSELoss()
        loss_fn = GANLoss()

        checkpoint_path = "../checkpoints/epoch_419.pth"  
        start_epoch = 0
        if os.path.exists(checkpoint_path):
            start_epoch = load_checkpoint(checkpoint_path, generator, discriminator, optimizer_g, optimizer_d, device)

        highest_psnr = 0.0
        highest_ssim = 0.0

#     # Training loop -> done successfully
        num_epochs = config['train']['num_epochs']

        #epoch running -> done successfully
        for epoch in range(start_epoch,num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            psnr_vals = []
            ssim_vals = []
            for i, (real_imgs, target_imgs) in enumerate(dataloader):
            
                real_imgs, target_imgs = real_imgs.to(device), target_imgs.to(device)
                # print(f"Batch {i} - Input Shape: {real_imgs.shape}, Target Shape: {target_imgs.shape}")
            
            # Train Discriminator
                optimizer_d.zero_grad()
                real_validity = discriminator(target_imgs)
                real_validity = F.interpolate(real_validity, size=real_imgs.shape[2:], mode="bilinear", align_corners=False)
                fake_imgs = generator(real_imgs).detach()
                fake_imgs = F.interpolate(fake_imgs, size=real_imgs.shape[2:], mode="bilinear", align_corners=False)
                # print(f"Fake Image Shape: {fake_imgs.shape}")

                fake_validity = discriminator(fake_imgs)
                fake_validity = F.interpolate(fake_validity, size=real_imgs.shape[2:], mode="bilinear", align_corners=False)
                d_loss = loss_fn.discriminator_loss(real_validity, fake_validity)
                d_loss.backward()
                optimizer_d.step()

                optimizer_g.zero_grad()
                fake_imgs = generator(real_imgs)
                fake_imgs = F.interpolate(fake_imgs, size=real_imgs.shape[2:], mode="bilinear", align_corners=False)
                fake_validity = discriminator(fake_imgs)
                fake_validity = F.interpolate(fake_validity, size=real_imgs.shape[2:], mode="bilinear", align_corners=False)
                g_loss = loss_fn.generator_loss(fake_validity, fake_imgs, target_imgs)
                g_loss.backward()
                optimizer_g.step()

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()

            # if i % 10 == 0:
            #     # wandb.log({"D_loss": d_loss.item(), "G_loss": g_loss.item()})
                # torchvision.utils.save_image(fake_imgs, f"../outputs/epoch_{epoch}_iter_{i}.png")
                torchvision.utils.save_image(fake_imgs, f"D:\\outputes/epoch_{epoch}_iter_{i}.png")
                print(f"Epoch {epoch}, Iter {i}, G_Loss: {g_loss.item()}, D_Loss: {d_loss.item()}")
                
                batch_psnr, batch_ssim = calculate_metrics(target_imgs, fake_imgs)
                psnr_vals.append(batch_psnr)
                ssim_vals.append(batch_ssim)
            epoch_psnr = sum(psnr_vals) / len(psnr_vals)
            epoch_ssim = sum(ssim_vals) / len(ssim_vals)

            if epoch_psnr > highest_psnr:
                highest_psnr = epoch_psnr
            if epoch_ssim > highest_ssim:
                highest_ssim = epoch_ssim

            # psnr, ssim = calculate_metrics(target_imgs, fake_imgs)
            display_metrics(epoch, epoch_psnr, epoch_ssim)
            log_losses(epoch, i, epoch_g_loss / len(dataloader), epoch_d_loss / len(dataloader))

            save_checkpoint(generator, discriminator, epoch, optimizer_g, optimizer_d)
        print(f"Highest PSNR: {highest_psnr:.2f}")
        print(f"Highest SSIM: {highest_ssim:.4f}")

if __name__ == '__main__':
    train()
