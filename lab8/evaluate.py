
import torch

from rbm import RBM
from dataset import test_loader


# Create RBM
rbm = RBM(n_visible=784, n_hidden=256)


# Load trained weights
checkpoint = torch.load(
    "outputs/rbm_mnist.pth",
    weights_only=True
)

rbm.W = checkpoint["W"]
rbm.visible_bias = checkpoint["visible_bias"]
rbm.hidden_bias = checkpoint["hidden_bias"]


# Evaluate on test dataset
total_error = 0

with torch.no_grad():

    for images, labels in test_loader:

        # Flatten images
        visible = images.view(images.shape[0], 784)

        # Binarize images
        visible = (visible > 0.5).float()

        # Encode and decode
        hidden_probs = rbm.hidden_probabilities(visible)

        reconstructed_probs = rbm.visible_probabilities(
            hidden_probs
        )

        # Calculate reconstruction error
        error = torch.mean(
            (visible - reconstructed_probs) ** 2
        )

        total_error += error.item()


# Average test reconstruction error
average_error = total_error / len(test_loader)

print("Test Reconstruction Error:", average_error)
