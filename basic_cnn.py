import torch
import torch.nn as nn

# Define the CNN model - this is not a super useful model in terms of classification, but I want to make sure I am indeed
# extracting the right embedding layer.
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) #should preserve shape
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 128)  # Assuming input image size is (64, 64, 3)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # Output layer with 10 classes

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.pool1(self.relu1(self.conv1(x)))
        print(x.shape) # should be 32 x 112 x 112 as input is 3 x 224 x 224. 
        x = self.pool2(self.relu2(self.conv2(x)))
        print(x.shape) # should be 64 x 56 x 56
        x = self.pool3(self.relu3(self.conv3(x)))
        print(x.shape) # should be 128 x 28 x 28

        # Flatten the output for fully connected layers
        print("x.shape after last maxpool2d is", x.shape)
        x = x.view(-1, 128 * 28 * 28)
        print(x.shape)

        # Forward pass through fully connected layers
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)

        return x