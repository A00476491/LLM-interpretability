import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, l1_lambda=0, tied_weights=0):
        super(SparseAutoencoder, self).__init__()
        self.l1_lambda = l1_lambda
        self.tied_weights = bool(tied_weights)
        self.mse_loss_fn = nn.MSELoss()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        if not self.tied_weights:
            self.W_dec = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(hidden_dim, input_dim)
                )
            )
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, x):
        x = x - self.b_dec
        z = self.encoder(x)
        x_hat = self.decode(z)
        return x_hat, z
    
    def decode(self, z):
        if self.tied_weights:
            x_hat = F.linear(z, self.encoder[0].weight.T, self.b_dec)
        else:
            x_hat = z @ self.W_dec  + self.b_dec
        return x_hat

    def loss(self, x, x_hat, z):
        mse_loss = self.mse_loss_fn(x_hat, x)
        l1_loss = torch.norm(z, 1)
        l0_loss = torch.count_nonzero(z).float()
        total_loss = mse_loss + self.l1_lambda * l1_loss
        return total_loss, mse_loss, l1_loss / z.shape[0], l0_loss / z.shape[0]

if __name__ == '__main__':

    sae_model = SparseAutoencoder(input_dim=896, hidden_dim=896*20).cuda()
    sae_model.load_state_dict(torch.load('./model/20250403-041718/best_model.pth'))
    print(sae_model)

    dummy_input = torch.rand(1, 896)
    with torch.no_grad():
        output, z = sae_model(dummy_input.cuda())

    print(dummy_input[0, :10])
    print(output[0, :10])