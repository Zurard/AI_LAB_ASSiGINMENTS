import torch
import matplotlib.pyplot as plt

from rbm import RBM
from dataset import test_loader


# Create RBM
rbm = RBM(n_visible=784, n_hidden=256)


# Load trained model
checkpoint = torch.load(
    "outputs/rbm_mnist.pth",
    weights_only=True
)

rbm.W = checkpoint["W"]
rbm.visible_bias = checkpoint["visible_bias"]
rbm.hidden_bias = checkpoint["hidden_bias"]


# Load test images
images, labels = next(iter(test_loader))

# Select 8 images
images = images[:8]
labels = labels[:8]

# Flatten and binarize
visible = images.view(8, 784)
visible = (visible > 0.5).float()


# Reconstruction
with torch.no_grad():

    hidden_probs = rbm.hidden_probabilities(visible)

    reconstructed_probs = rbm.visible_probabilities(
        hidden_probs
    )


# Visualization
fig, axes = plt.subplots(2, 8, figsize=(12, 4))

for i in range(8):

    # Original image
    axes[0, i].imshow(
        images[i].squeeze(),
        cmap="gray"
    )

    axes[0, i].set_title(
        str(labels[i].item())
    )

    axes[0, i].axis("off")


    # Reconstructed image
    axes[1, i].imshow(
        reconstructed_probs[i].view(28, 28),
        cmap="gray"
    )

    axes[1, i].axis("off")


axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Reconstructed")

plt.suptitle("MNIST Test Reconstruction")

plt.tight_layout()

plt.savefig(
    "outputs/test_reconstruction.png",
    dpi=300
)

plt.show()