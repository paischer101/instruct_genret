import torch.nn as nn
import torch.nn.functional as F
from .layers import MLP, QuantizationLayer

class RQVAE(nn.Module):
    # jiacheng's
    def __init__(self, input_size, hidden_sizes, latent_size, num_levels, codebook_size, dropout, use_normalization):
        super(RQVAE, self).__init__()
        self.encoder = MLP(input_size, hidden_sizes, latent_size, dropout=dropout, use_normalization=use_normalization)
        self.quantization_layer = QuantizationLayer(num_levels, codebook_size, latent_size)
        hidden_sizes.reverse()
        self.decoder = MLP(latent_size, hidden_sizes, input_size, dropout=dropout, use_normalization=use_normalization)

    def forward(self, x, return_flops=False):
        encoded = self.encoder(x)
        quantized ,r, e, z_hat, counts, flops_loss = self.quantization_layer(encoded)
        decoded = self.decoder(z_hat)
        if return_flops:
            return decoded, r, e, counts, quantized, flops_loss
        return decoded, r, e, counts, quantized
    
    def encode(self, x):
        encoded =  self.encoder(x)
        quantized, _, _, _, _, _ = self.quantization_layer(encoded)
        return quantized.detach().cpu().numpy()

    def generate_codebook(self, x, device):
        encoded = self.encoder(x)
        self.quantization_layer.generate_codebook(encoded, device)