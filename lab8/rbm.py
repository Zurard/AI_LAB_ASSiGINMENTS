# Our RBM has:
# - 784 visible neurons — one for each pixel.
# - 64 hidden neurons — learn patterns from the pixels.
# - 50,176 weights — connections between visible and hidden neurons.
# - 784 visible biases.
# - 64 hidden biases.

import matplotlib.pyplot as plt
from dataset import train_loader
import torch 
import time 

class RBM : 
    def __init__(self , n_visible = 784 , n_hidden = 64):
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        # initializing the weight 
        self.W = torch.randn(
            n_visible,
            n_hidden,
        ) * 0.01
        
        
        self.visible_bias = torch.zeros(n_visible)
        self.hidden_bias = torch.zeros(n_hidden)
    
    
    # Sigmoid converts any numerical value into a probability between 0 and 1.
    def sigmoid(self , x ):
        return torch.sigmoid(x)
        
    def hidden_probabilities(self , visible):
        hidden_input = torch.matmul( visible, self.W ) + self.hidden_bias
        return self.sigmoid(hidden_input)
        
    def sample_hidden(self , visible ):
        hidden_probs = self.hidden_probabilities(visible)
        
        hidden_states = torch.bernoulli(hidden_probs)
        
        return hidden_probs , hidden_states 
    
        
    def visible_probabilities(self, hidden):

        visible_input = torch.matmul(hidden,self.W.T) + self.visible_bias

        # Convert weighted input into pixel probabilities
        visible_probs = self.sigmoid(visible_input)

        return visible_probs
    
    def sample_visible(self , hidden ):
        visible_probs = self.visible_probabilities(hidden)
        visible_states = torch.bernoulli(visible_probs)
        
        return visible_probs , visible_states



    def gibbsStep(self , visible): 
        hidden_probs, hidden_states = self.sample_hidden(visible)
        visible_probs, visible_states = self.sample_visible(hidden_states)
        
        return visible_probs , visible_states
    
    
    def train_step(self, visible, learning_rate=0.01, k=5):

        # Positive phase
        positive_hidden_probs, positive_hidden_states = self.sample_hidden(visible)

        # Start negative phase from positive hidden states
        hidden_states = positive_hidden_states

        # Perform k Gibbs sampling steps
        for step in range(k):

            # Sample visible units
            negative_visible_probs, negative_visible_states = self.sample_visible(
                hidden_states
            )

            # Sample hidden units from reconstructed visible units
            negative_hidden_probs, negative_hidden_states = self.sample_hidden(
                negative_visible_states
            )

            # Continue chain
            hidden_states = negative_hidden_states

        # Positive association
        positive_association = torch.matmul(
            visible.T, positive_hidden_probs
        )

        # Negative association
        negative_association = torch.matmul(
            negative_visible_states.T, negative_hidden_probs
        )

        # Update weights
        self.W += learning_rate * (
            positive_association - negative_association
        ) / visible.shape[0]

        # Update visible bias
        self.visible_bias += learning_rate * torch.mean(
            visible - negative_visible_states, dim=0
        )

        # Update hidden bias
        self.hidden_bias += learning_rate * torch.mean(
            positive_hidden_probs - negative_hidden_probs, dim=0
        )

        # Reconstruction error
        reconstruction_error = torch.mean(
            (visible - negative_visible_probs) ** 2
        )

        return reconstruction_error.item()
        
        
        
        
if __name__ == "__main__": 
    rbm = RBM()
   
    epochs = 15
    learning_rate = 0.01

    for epoch in range(epochs):
        total_error = 0

        start_time = time.time()

        for images, labels in train_loader:

            # Flatten images into 784 pixels
            visible = images.view(images.shape[0], 784)

            # Convert pixels into binary values
            visible = (visible > 0.5).float()

            # Train the RBM on this batch
            error = rbm.train_step(
                visible,
                learning_rate=learning_rate,
                k = 5
            )

            total_error += error

        average_error = total_error / len(train_loader)

        elapsed_time = time.time() - start_time

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Error: {average_error:.6f} "
            f"- Time: {elapsed_time:.2f}s"
        )
        
    torch.save(
    {
        "W": rbm.W,
        "visible_bias": rbm.visible_bias,
        "hidden_bias": rbm.hidden_bias,
    },
    "outputs/rbm_mnist.pth"
    )

    print("Model saved successfully!")

   
   
   
   
   
    
    # visible = torch.bernoulli(torch.rand(1, 784))

    # for i in range(10):
    #     visible_probs, visible_states = rbm.gibbssStep(visible)
            
    #     visible = visible_states
    #     print(f"Step {i + 1} : {visible.shape}")        
    
    # # calc hidden probs 
    # hidden_probs , hidden_states = rbm.sample_hidden(visible)
    
    # print(hidden_states.shape , hidden_probs.shape)
    
    # visible_probs, reconstructed_visible = rbm.sample_visible(
    #     hidden_states
    # )

    # print(visible_probs.shape , reconstructed_visible.shape)
      
    # print(rbm.W.shape)
    
    # print(rbm.hidden_bias.shape)
    # print(rbm.visible_bias.shape)
        
        
        