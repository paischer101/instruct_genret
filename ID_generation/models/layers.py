from torch import nn
import torch
from sklearn.cluster import KMeans
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size, dropout=0.0, use_normalization=False):
        """
        :param input_size: input embedding size
        :param hidden_sizes: a list of hidden size
        :param latent_size: the final encoded size
        """
        super(MLP, self).__init__()
        mlp_modules = []
        hidden_sizes = [input_size] + hidden_sizes + [latent_size]
        for idx, (input_size, output_size) in enumerate(
            zip(hidden_sizes[:-1], hidden_sizes[1:])
        ):
            mlp_modules.append(nn.Linear(input_size, output_size))
            if use_normalization:
                mlp_modules.append(nn.LayerNorm(normalized_shape=(output_size)))

            activation_func = nn.ReLU()
            if idx != len(hidden_sizes) - 2:
                mlp_modules.append(activation_func)
            
            if dropout > 0:
                mlp_modules.append(nn.Dropout(p=dropout))
            else:
                raise NotImplementedError("Currently only implemented for Dropout and BatchNorm")
        self.mlp = nn.Sequential(*mlp_modules)
    def forward(self, x):
        return self.mlp(x)
    
class QuantizationLayer(nn.Module):
    # codebook_size can be either an int (meaning we use same codebook_size for all levels), 
    # or a list of int to specify the codebook_size for each codebook level.
    def __init__(self, num_levels, codebook_size, latent_size):
        # codebook size means the number of unique code.
        # num_levels: how many codebooks.
        # latent_size: each code vector dimension.
        super(QuantizationLayer, self).__init__()
        self.num_levels = num_levels
        self.latent_size = latent_size
        
        # Check if codebook_size is an int and convert it to a list of the same size for each level
        if isinstance(codebook_size, int):
            self.codebook_sizes = [codebook_size] * num_levels
        elif isinstance(codebook_size, list) and len(codebook_size) == num_levels:
            self.codebook_sizes = codebook_size
        else:
            raise ValueError("codebook_size must be an int or a list of int with length equal to num_levels")

        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.zeros(self.codebook_sizes[l], latent_size))
            for l in range(num_levels)
        ])
        
    def forward(self, x):
        batch_size, _ = x.shape
        output = torch.empty(batch_size, self.num_levels, dtype=torch.long, device=x.device)
        # x -> batch_size * 32
        # codebook -> codebook_size*32
        r = []
        e = []
        count = 0
        z_hat = x
        flops_loss = 0
        for l in range(self.num_levels):
            r.append(x)
            x = x.unsqueeze(1)  # (batch_size, 1, latent_size)
            distances = torch.norm(x - self.codebooks[l].unsqueeze(0), dim=-1)  # Calculate distances to codebook entries
            flops_dist = flops_regularizer(distances)
            flops_loss += flops_dist
            indices = torch.argmin(distances, dim=-1)
            quantization = self.codebooks[l].unsqueeze(0)[:,indices,:].squeeze(0)  # Perform quantization
            output[:, l] = indices
            x = x.squeeze(1)
            counts = torch.bincount(indices, minlength=self.codebook_sizes[l])
            # Find codebook entries with usage < 1.0
            low_usage_indices = torch.where(counts < 1.0)[0]
            count += len(low_usage_indices)
            x = x - quantization
            e.append(quantization)
        
        e = torch.stack(e, dim=1)
        r = torch.stack(r, dim=1)
        z_hat = z_hat + (e.detach().sum(dim=1) - z_hat.detach())
        return output, r, e, z_hat, count, flops_loss
    
    def generate_codebook(self, x, device):
        """
        This is the method to initialize the codebook
        """
        for l in range(self.num_levels):
            kmeans = KMeans(n_clusters=self.codebook_sizes[l], n_init='auto').fit(x.detach().cpu().numpy())
            # liu: every level of codebook is initialized with the same size.
            self.codebooks[l].data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=device)
            x = x.unsqueeze(1)  # (batch_size, 1, latent_size)
            distances = torch.norm(x - self.codebooks[l].unsqueeze(0), dim=-1)  # Calculate distances to codebook entries
            indices = torch.argmin(distances, dim=-1)  # (batch_size,)
            quantization = self.codebooks[l].unsqueeze(0)[:,indices,:].squeeze(0)  # Perform quantization
            x = x.squeeze(1)
            x = x - quantization



def flops_regularizer(
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Embeddings is of shape (batch_size, num_clusters)
    - Find the mean of every cluster weight
    - Minimize the sum of squares of this.
    """
    l1_norm_col = torch.mean(embeddings, dim=0)
    mean_flops_sur = torch.sum(l1_norm_col * l1_norm_col)
    return mean_flops_sur