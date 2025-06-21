import torch
import torch.nn as nn
from utils.value_network.modules import ResNet, Bottleneck, Normalize, Unnormalize
from utils.value_network.modules import SingleBVPNet

class LiDARValueNN(nn.Module):
    """LiDAR-conditioned value network. Takes as input (x, y, th, dst_dxdy_max, dst_dth_max, lidar)."""
    def __init__(
            self,
            input_means: torch.Tensor,
            input_stds: torch.Tensor,
            output_mean: torch.Tensor,
            output_std: torch.Tensor,
            input_dim: int = 105,
            activation: str = 'sine',
    ):
        """Initializes a LiDAR-conditioned value network.
        
        Args:
            input_means: A torch tensor with shape [input_dim].
            input_stds: A torch tensor with shape [input_dim].
            output_mean: A torch tensor with shape [1].
            output_std: A torch tensor with shape [1].
            input_dim: The size of the input.
        """
        super().__init__()
        self.input_normalizer = Normalize(input_means, input_stds)
        self.output_unnormalizer = Unnormalize(output_mean, output_std)
        self.mlp = SingleBVPNet(in_features=input_dim, out_features=1, type=activation, mode='mlp', final_layer_factor=1., hidden_features=512, num_hidden_layers=3)

    def forward(
            self,
            inputs: torch.Tensor
    ):
        """Computes a forward pass through the network.
        
        Args:
            inputs: A torch tensor with shape [batch_size, input_dim].

        Returns:
            values: A torch tensor with shape [batch_size].
        """
        return self.output_unnormalizer(self.mlp(self.input_normalizer(inputs))).squeeze(-1)

class DepthImageValueNN(nn.Module):
    def __init__(self,
                 latent_size,
                 state_dim):
        super().__init__()
        self.encoder = ResNet(Bottleneck, [2, 2, 2, 2], latent_size)
        self.mlp = SingleBVPNet(in_features=latent_size+state_dim, out_features=1, type='sine', mode='mlp', final_layer_factor=1., hidden_features=512, num_hidden_layers=3)

    """
    observations: N x H x W
    states: N x M x state_dim
    returns values: N x M
    """
    def forward(self, observations, states):
        N, M, _ = states.shape
        latents = self.encoder(observations.unsqueeze(1)) # N x latent_size
        latents_and_states = torch.cat((latents.unsqueeze(1).expand(N, M, -1), states), dim=-1)
        return self.mlp(latents_and_states.view(N*M, -1)).view(N, M)