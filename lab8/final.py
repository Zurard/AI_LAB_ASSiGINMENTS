
import torch
import matplotlib.pyplot as plt

from rbm import RBM


# Configuration
NUM_SAMPLES = 10
GIBBS_STEPS = 500


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


# Start with random binary pixels
visible = torch.bernoulli(
    torch.rand(NUM_SAMPLES, 784)
)


# Gibbs sampling
with torch.no_grad():

    for step in range(GIBBS_STEPS):

        # Sample hidden units
        hidden_probs, hidden_states = rbm.sample_hidden(
            visible
        )

        # Sample visible units
        visible_probs, visible_states = rbm.sample_visible(
            hidden_states
        )

        # Continue the sampling chain
        visible = visible_states


# Convert generated samples into images
generated_images = visible.view(
    NUM_SAMPLES, 28, 28
)


# Display generated images
fig, axes = plt.subplots(
    2, 5,
    figsize=(10, 5)
)

for i, ax in enumerate(axes.flat):

    ax.imshow(
        generated_images[i].cpu(),
        cmap="gray"
    )

    ax.axis("off")


plt.suptitle(
    "RBM Generated Images from Random Initialization"
)

plt.tight_layout()


# Save image grid
plt.savefig(
    "outputs/final_generated_images.png",
    dpi=300,
    bbox_inches="tight"
)

print("Generated images saved successfully!")

plt.show()
