# import torch
# from src.models.generator import Generator
# from torchvision.utils import save_image
# from src.dataloader import get_test_loader

# # Load the generator
# generator = Generator().to("cpu")
# generator.load_state_dict(torch.load("outputs/generator.pth"))

# # Load test data
# test_loader = get_test_loader("configs/config.yaml")

# # Inference
# for i, real_imgs in enumerate(test_loader):
#     real_imgs = real_imgs.to("cpu")
#     enhanced_imgs = generator(real_imgs)
#     save_image(enhanced_imgs, f"outputs/enhanced_{i}.png")

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.generator import Generator
from utils import calculate_metrics, display_metrics
import torchvision

def load_checkpoint(checkpoint_path, generator):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    print(f"Checkpoint loaded: {checkpoint_path}")

def process_single_image(image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0).to(device)


def test_single_image():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    checkpoint_path = "C:\\Users\\yjyas\\Projects\\checkpoints\\epoch_99.pth"
    load_checkpoint(checkpoint_path, generator)
    generator.eval()

    input_image_path = "C:\\Users\\yjyas\\Projects\\dataset\\test\\input\\232.jpg"  
    output_image_path = "C:\\Users\\yjyas\\Projects\\dataset\\test\\input\\enhanced_image.png"

    tran = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    real_image = process_single_image(input_image_path, tran, device)

    with torch.no_grad():
        enhanced_image = generator(real_image)

    torchvision.utils.save_image(enhanced_image, output_image_path)
    print(f"Enhanced image saved at: {output_image_path}")


if __name__ == '__main__':
    test_single_image()
