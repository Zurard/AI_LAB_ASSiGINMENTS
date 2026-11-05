# will laod the dataset in this file 

import torch
import torchvision
import matplotlib as plt 

from torchvision import datasets , transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

# downlaodiing the datatset
train_dataset = datasets.MNIST(
    root= "./data",
    train=True,
    download=True,
    transform = transform
)


# load the MNIST test dataset 
test_dataset = datasets.MNIST(
    root= "./data",
    train=False ,
    download=True,
    transform = transform
)


# now we need to create batches of the dataset 
train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=64
)

test_loader = DataLoader(
    test_dataset,
    shuffle=True,
    batch_size=64
)


print("Training images:" , len(train_dataset))
print("testing images:" , len(test_dataset))


images , labels = next(iter(train_loader))
# batch image shape 
print(images.shape)
# batch label shape 
print(labels.shape)



# now we need to flatten the image  
images = images.view(images.shape[0] , -1)

# covert grey scale into binary values 
images = (images > 0.50).float()


print(images.shape)
print(torch.unique(images)) 