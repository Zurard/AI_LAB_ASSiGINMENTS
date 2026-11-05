import torch
import matplotlib.pyplot as plt

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


# Load real MNIST images
images, labels = next(iter(test_loader))

# Select 10 images
visible = images[:10].view(10, 784)

# Binarize images
visible = (visible > 0.5).float()


# Perform Gibbs sampling
for step in range(500):

    hidden_probs, hidden_states = rbm.sample_hidden(visible)

    visible_probs, visible_states = rbm.sample_visible(hidden_states)

    visible = visible_states


# Display generated images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):

    image = visible[i].view(28, 28)

    ax.imshow(image, cmap="gray")
    ax.axis("off")

plt.suptitle("RBM Generated Images from MNIST Chains")
plt.tight_layout()

plt.savefig("outputs/generated_digits.png")

plt.show()