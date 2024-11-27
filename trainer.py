pip install torch torchvision transformers datasets accelerate

#Vamos a hacer todos los imports necesarios

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

#Configuramos el transformer

d_model = 512  # Dimensión de las características para texto e imágenes
latent_dim = 128  # Dimensión latente del VQ-VAE

#Definimos el VQ-VAE

class VQVAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, kernel_size=4, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),         # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),           # 32x32
            nn.Tanh(),
        )

    def forward(self, x):
        latents = self.encoder(x)
        recon = self.decoder(latents)
        return latents, recon

#Definimos el transformer que va a pasar el texto a imagen
class TextToImageTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.text_projection = nn.Linear(256, d_model)  # Proyecta texto al tamaño d_model
        self.latent_projection = nn.Linear(latent_dim, d_model)  # Proyecta latentes al tamaño d_model
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.output_projection = nn.Linear(d_model, latent_dim)  # Proyecta salida al espacio latente

    def forward(self, text_features, image_latents):
        # Proyectar texto e imágenes a d_model
        text_features = self.text_projection(text_features)  # (batch_size, seq_len, d_model)
        image_latents = self.latent_projection(image_latents)  # (batch_size, 1, d_model)

        # Pasar por el transformer
        transformer_output = self.transformer(
            src=text_features,
            tgt=image_latents
        )

        # Proyectar salida al espacio latente original
        predicted_latents = self.output_projection(transformer_output)
        return predicted_latents

#Inicializamos los modelos
vqvae = VQVAE(latent_dim)
text_to_image = TextToImageTransformer(d_model=d_model)

#Cagramos el dataset y el transform para ajustar las imágenes
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

#Optimización
optimizer = optim.Adam(list(vqvae.parameters()) + list(text_to_image.parameters()), lr=1e-3)

#Entrenamiento
for epoch in range(10):
    for images, _ in data_loader:
        optimizer.zero_grad()

        # Codificar imágenes a latentes
        latents, recon_images = vqvae(images)  # latents: (batch_size, latent_dim, h, w)
        latents = latents.mean(dim=[2, 3])  # Promediar para obtener (batch_size, latent_dim)

        # Procesar texto
        batch_size = images.size(0)
        text = ["car"] * batch_size  # Texto repetido para todo el lote
        text_features = process_text(text)  # Dimensiones: (batch_size, seq_len, 256)

        # Asegurar que latentes tengan dimensión de secuencia
        latents = latents.unsqueeze(1)  # De (batch_size, latent_dim) a (batch_size, 1, latent_dim)

        # Generar latentes desde texto
        predicted_latents = text_to_image(text_features, latents)  # (batch_size, seq_len, latent_dim)

        # Reconstruir imágenes
        predicted_latents = predicted_latents.squeeze(1) 
        # Expandir a (batch_size, latent_dim, 8, 8) para que coincida con la salida del decoder
        recon_from_text = vqvae.decoder(predicted_latents.unsqueeze(2).unsqueeze(3))  

        # Interpolar recon_from_text a 32x32
        recon_from_text = F.interpolate(recon_from_text, size=(32, 32), mode='bilinear', align_corners=False)

        # Calcular pérdida
        loss = F.mse_loss(recon_images, images) + F.mse_loss(recon_from_text, images)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

#Guardamos el modelo

def save_model(model, optimizer, epoch, loss, filename='modelo_entrenado.pth'):
    torch.save({
        'epoch': epoch,  # El número de la última época
        'model_state_dict': model.state_dict(),  # Los pesos del modelo
        'optimizer_state_dict': optimizer.state_dict(),  # El estado del optimizador
        'loss': loss,  # La última pérdida calculada (opcional)
    }, filename)
    print(f'Modelo guardado como {filename}')

# Llamar a la función para guardar después de entrenar
save_model(model, optimizer, epoch, loss)

