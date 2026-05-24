import numpy as np
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__ (self, input_dim = 256, hidden_dim = 1024):

        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias = True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias = True)
        self.relu = nn.ReLU()

        # The point of the spareseautoencoder is to take the activations we have collected from the RL network and then encode them into a longer dimension forcing 
        # only a few neurons to be active (topk) and then compress them back into the original dimension. The goal is for it to train over time so that the (input - reoncstructed)^2
        # is generally low because that means sparse features capture important information. ReLU helps with this by making sparse activations, only a few are positive to see more
        # clearly what neurons actually fired.

    def encode(self, activations, k = 32):

        encoded_representation = self.relu((self.encoder(activations)))
        encoded_representation = self.batch_topk(encoded_representation, k)

        return encoded_representation

        # This is the encode function where we scale the inpuut dimension into a higher dimension and then put it through ReLU to see which neurons are actually firing, and then
        # taking the top k (32) of those in order to see stronger activations

    def decode(self, encoded_representation):

        return self.decoder(encoded_representation)

        # This is just the decoding function that puts the encoded representation back into the original dimension size.

    def forward_pass(self, activations):

        encoded_representation = self.encode(activations)
        reconstructed = self.decode(encoded_representation)

        return reconstructed, encoded_representation

        # This is just the forward_pass where we run both encode and deocde so we can get the encoded_representation which is the scaled up version of the activations
        # and we also get reconstructed which is just the encoded_representation put through the decode function

    def batch_topk(self, encoded_representation, k):

        # Find the top k largest activations and their indices
        topk_values, topk_indices = torch.topk(encoded_representation, k, dim=-1)

        # Create a mask of zeros, then put 1s at the top-k positions
        mask = torch.zeros_like(encoded_representation)
        mask.scatter_(-1, topk_indices, 1.0)

        # Keep top-k activations, zero out the rest
        # The weird math with .detach() is a "straight-through estimator" trick —
        # it zeros out non-top-k values in the forward pass, but lets gradients
        # flow through all neurons during backprop so the network can still learn
        # which neurons *should* be in the top-k
        return encoded_representation * mask + encoded_representation.detach() * (1 - mask) - encoded_representation.detach() * (1 - mask)

    def loss(self, activations):

        reconstructed, encoded = self.forward_pass(activations)
        reconstruction_loss = ((activations - reconstructed)**2).mean()

        return reconstruction_loss

        # This is the loss function and it is basically the mean squared difference between the actual activations and the reconstructed version we have after running
        # them through the forward_pass because the goal of the SAE is to see that even after making doing sparse activations, can the output look the same as the input
        # because it proves the sparsity is working

        # Without sparsity, the SAE could just learn identity (copy input → output). Boring, useless. With sparsity (TopK), 
        # it's forced to compress the information into a small number of active features. If it can still reconstruct well, 
        # those features must be disentangled — each one encoding something specific rather than a messy superposition.

    def train_sae(self, activations, epochs = 100, lr = 1e-4, batch_size = 64):

        print("Training started...")

        optimizer = torch.optim.Adam(self.parameters(), lr = lr)

        # This is a gradient descent optimizer and it is a fancier way to do gradient descent that is not 
        # just the generic gradient step as this specific one tracks momentum and adapts the learning rate.

        for epoch in range(epochs):

            indices = torch.randperm(len(activations))
            activations_shuffled = activations[indices]

            # This is the main training part where we train the SAE based on the activations we have from the RL policy network. This indices variable just makes
            # the permutation of the activation values random so that the SAE does not recognize the order of the activations

            epoch_loss = 0
            num_batches = 0

            # Epoch loss is the total cumulative loss across all the batches in a single epoch and num batches just counts how many batches you processed
            # so at the end you can calculate the avg loss.

            for i in range(0, len(activations), batch_size):
                batch = activations_shuffled[i:i+batch_size]

                # This for loop just makes batches of 64 within the activations list and then we calculate the loss based on our earlier method to see how
                # our SAE does for that specific set of actions. 

                loss = self.loss(batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Gradients always accumulate so before doing the backpropogation for a specific batch we have to reset the optimizer by clearing it and then
                # we do the backprop based on the loss we calculated and then we step with the optiimzier so it can continue the gradient descent.

                epoch_loss += loss.item()
                num_batches += 1

                # Then we just add the current loss of the batch to the total epoch loss and then add plus one to the num_batches

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, avg loss: {epoch_loss/num_batches:.4f}")

                # This just prints the avg loss every 10 epochs

activations = torch.load('/Users/maroonferrari/Personal Project/RL-Interpretability/agents/activations.pt')
print(f"Shape: {activations.shape}")

sae = SparseAutoencoder()
sae.train_sae(activations)
torch.save(sae.state_dict(), 'sae_trained.pt')