import numpy as np
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__ (self, input_dim = 256, hidden_dim = 1024):

        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias = True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias = True)
        self.relu = nn.ReLU()

    def encode(self, activations, k = 32):

        encoded_representation = self.relu((self.encoder(activations)))
        encoded_representation = self.batch_topk(encoded_representation, k)

        return encoded_representation

    def decode(self, encoded_representation):

        return self.decoder(encoded_representation)

    def forward_pass(self, activations):

        encoded_representation = self.encode(activations)
        reconstructed = self.decode(encoded_representation)

        return reconstructed, encoded_representation

    def batch_topk(self, encoded_representation, k):

        topk_values, topk_indices = torch.topk(encoded_representation, k, dim=-1)
        mask = torch.zeros_like(encoded_representation)
        mask.scatter_(-1, topk_indices, 1.0)

        return encoded_representation * mask + encoded_representation.detach() * (1 - mask) - encoded_representation.detach() * (1 - mask)

    def loss(self, activations):

        reconstructed, encoded = self.forward_pass(activations)
        reconstruction_loss = ((activations - reconstructed)**2).mean()

        return reconstruction_loss

    def train_sae(self, activations, epochs = 100, lr = 1e-4, batch_size = 64):

        print("Training started...")

        optimizer = torch.optim.Adam(self.parameters(), lr = lr)

        for epoch in range(epochs):

            indices = torch.randperm(len(activations))
            activations_shuffled = activations[indices]

            epoch_loss = 0
            num_batches = 0

            for i in range(0, len(activations), batch_size):
                batch = activations_shuffled[i:i+batch_size]

                loss = self.loss(batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, avg loss: {epoch_loss/num_batches:.4f}")

activations = torch.load('/Users/maroonferrari/Personal Project/RL-Interpretability/agents/activations.pt')
print(f"Shape: {activations.shape}")

sae = SparseAutoencoder()
sae.train_sae(activations)
torch.save(sae.state_dict(), 'sae_trained.pt')