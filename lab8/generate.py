import torch 
import matplotlib.pyplot as plt

from rbm import RBM
from dataset import test_loader


rbm = RBM(n_visible=764 , n_hidden=256  )


# loading the model 
checkpoint =   torch.load("outputs/rbm_mnist.pth" , weights_only=True)

rbm.W = checkpoint["W"]
rbm.visible_bias = checkpoint["visible_bias"]
rbm.hidden_bias = checkpoint["hidden_bias"]


# Load one real MNIST image
images, labels = next(iter(test_loader))

image = images[0]

# Flatten image
visible = image.view(1, 784)

# Binarize image
visible = (visible > 0.5).float()


# Encode the original image into hidden probabilities
hidden_probs = rbm.hidden_probabilities(visible)

# Decode hidden probabilities directly into visible probabilities
reconstructed_probs = rbm.visible_probabilities(hidden_probs)

# Convert to image format
reconstructed_image = reconstructed_probs.view(28, 28)

# Display images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Original Digit: {labels[0].item()}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image.detach(), cmap="gray")
plt.title("Trained RBM Reconstruction")
plt.axis("off")

plt.tight_layout()
plt.show()