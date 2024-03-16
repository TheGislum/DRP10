import torch
import torch.nn as nn


class PosLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PosLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn((in_dim, out_dim)))
        self.bias = nn.Parameter(torch.zeros((out_dim,)))

    def forward(self, x):
        return torch.matmul(x, torch.exp(self.weight)) + self.bias


class BetaVAE(nn.Module):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        l_1: int = 128,
        beta: int = 4,
        loss_type: str = "mse",
    ) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.loss_type = loss_type

        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, l_1),
            nn.ReLU(),
            nn.BatchNorm1d(l_1),
            nn.Linear(l_1, int(l_1 / 2)),
            nn.ReLU(),
            nn.BatchNorm1d(int(l_1 / 2)),
            nn.Linear(int(l_1 / 2), int(l_1 / 4)),
            nn.ReLU(),
            nn.BatchNorm1d(int(l_1 / 4)),
        )

        # Latent vectors mu and sigma
        self.fc_mu = nn.Linear(int(l_1 / 4), latent_dim)
        self.fc_var = nn.Linear(int(l_1 / 4), latent_dim)

        # Build Decoder
        self.decoder = PosLinear(latent_dim, input_dim)

    def encode(self, input: torch.Tensor):
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: torch.Tensor):
        result = self.decoder(z)
        result = torch.relu(result)
        return result

    def forward(self, input: torch.Tensor):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), mu, log_var]

    def loss_function(self, recons, input, mu, log_var, kld_weight=1.0):
        if self.loss_type == "mse":
            recons_loss = nn.functional.mse_loss(recons, input, reduction="sum")
        elif self.loss_type == "bce":
            recons_loss = nn.functional.binary_cross_entropy(
                recons, input, reduction="sum"
            )
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )
        loss = recons_loss + self.beta * kld_weight * kld_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "KLD": kld_loss}


class pilot_BetaVAE(nn.Module):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list = None,
        beta: int = 4,
        loss_type: str = "mse",
    ) -> None:
        super(pilot_BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.loss_type = loss_type

        modules = []

        # Build Encoder
        in_channels = input_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.ReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.ReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], input_dim), nn.Sigmoid()
        )

    def encode(self, input: torch.Tensor):
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: torch.Tensor):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: torch.Tensor):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), mu, log_var]

    def loss_function(self, recons, input, mu, log_var, kld_weight=1.0):
        if self.loss_type == "mse":
            recons_loss = nn.functional.mse_loss(recons, input, reduction="sum")
        elif self.loss_type == "bce":
            recons_loss = nn.functional.binary_cross_entropy(
                recons, input, reduction="sum"
            )
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )
        loss = recons_loss + self.beta * kld_weight * kld_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "KLD": kld_loss}
