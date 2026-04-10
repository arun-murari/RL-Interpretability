import sys
import torch 
import numpy as np

sys.path.append('/Users/maroonferrari/Personal Project/RL-Interpretability/interpretability')
from sparseautoencoder import SparseAutoencoder

activations = torch.load('/Users/maroonferrari/Personal Project/RL-Interpretability/agents/activations.pt')
sae = SparseAutoencoder()
sae.load_state_dict(torch.load('/Users/maroonferrari/Personal Project/RL-Interpretability/interpretability/sae_trained.pt'))

with torch.no_grad():
    encoded = sae.encode(activations)

feature_activation_counts = (encoded > 0).sum(dim=0)
top_features = torch.argsort(feature_activation_counts, descending=True)[:10]

states = torch.load('/Users/maroonferrari/Personal Project/RL-Interpretability/agents/states.pt')

for i, feat_idx in enumerate(top_features[:10]):
    feature_values = encoded[:, feat_idx]
    top_samples = torch.argsort(feature_values, descending=True)[:20]
    top_states = states[top_samples]
    
    x_velocities = top_states[:, 1]  # column 1 is x-velocity
    print(f"Feature {feat_idx.item()}: mean x-vel = {x_velocities.mean():.2f}, std = {x_velocities.std():.2f}")

for feat_idx in range(1024):
    if (encoded[:, feat_idx] > 0).sum() < 100:
        continue 
    
    feature_values = encoded[:, feat_idx]
    top_samples = torch.argsort(feature_values, descending=True)[:20]
    top_states = states[top_samples]
    
    mean_x_vel = top_states[:, 1].mean().item()
    mean_z_pos = top_states[:, 0].mean().item()
    std_x_vel = top_states[:, 1].std().item()
    
    if mean_x_vel < 1.5 and std_x_vel < 0.5:
        print(f"SLOW Feature {feat_idx}: x-vel = {mean_x_vel:.2f}")

    if mean_z_pos < -0.5 or mean_z_pos > 0.3:
        print(f"TILT Feature {feat_idx}: z-pos = {mean_z_pos:.2f}")

for feat_idx in range(1024):
    if (encoded[:, feat_idx] > 0).sum() < 100:
        continue
    
    feature_values = encoded[:, feat_idx]
    top_samples = torch.argsort(feature_values, descending=True)[:20]
    top_states = states[top_samples]
    
    mean_z = top_states[:, 0].mean().item()
    std_z = top_states[:, 0].std().item()
    
    if mean_z > 0.2 and std_z < 0.3:
        print(f"UPRIGHT Feature {feat_idx}: z-pos = {mean_z:.2f}, std = {std_z:.2f}")

print(f"z-pos range: min = {states[:, 0].min():.2f}, max = {states[:, 0].max():.2f}, mean = {states[:, 0].mean():.2f}")

# Find features that care about a specific joint
for feat_idx in range(1024):
    if (encoded[:, feat_idx] > 0).sum() < 100:
        continue
    
    feature_values = encoded[:, feat_idx]
    top_samples = torch.argsort(feature_values, descending=True)[:20]
    top_states = states[top_samples]
    
    # Check joint 2 (first leg angle)
    mean_joint = top_states[:, 2].mean().item()
    std_joint = top_states[:, 2].std().item()
    
    if std_joint < 0.2:  # Very consistent joint angle
        print(f"JOINT Feature {feat_idx}: joint2 = {mean_joint:.2f}, std = {std_joint:.2f}")