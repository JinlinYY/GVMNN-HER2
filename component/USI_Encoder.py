import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_image_features(image_filenames):
    model = models.vit_l_32(weights='DEFAULT')
    model.fc = torch.nn.Identity()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model.train()
    model.to(device)

    image_features = []
    with torch.set_grad_enabled(False):  # Disable gradients
        for filename in image_filenames:
            image = Image.open(filename).convert('RGB')
            input_tensor = preprocess(image).unsqueeze(0).to(device)
            features = model(input_tensor)
            features = features.squeeze().cpu().detach().numpy()  # Squeeze to remove batch dimension
            image_features.append(features)

    return np.vstack(image_features)



