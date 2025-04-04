import torch
import torch.nn as nn
import torch.optim as optim

# Create a simple neural network with one layer
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer = nn.Linear(1, 1)  # A simple layer (1 input, 1 output)
        
    def forward(self, x):
        return self.layer(x)

# Instantiate the neural network
model = SimpleNN()

# Loss function (Mean Squared Error) and Optimizer (Stochastic Gradient Descent)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent with learning rate of 0.01

# Training data (let’s pretend we’re learning to add numbers)
inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # Input values
targets = torch.tensor([[2.0], [3.0], [4.0], [5.0]])  # Target values (input + 1)

# Training loop (we’ll run it 1000 times)
for epoch in range(2000):
    # Forward pass: Compute predicted y by passing x to the model
    predictions = model(inputs)  # Pass the inputs through the model (predict)
    
    # Compute the loss (difference between prediction and target)
    loss = criterion(predictions, targets)  # How wrong is our model?

    # Backward pass: Compute gradients of all weights in the model
    optimizer.zero_grad()  # Zero the gradients before backpropagation
    loss.backward()  # Autograd computes the gradients of the loss with respect to all weights

    # Update weights with the optimizer
    optimizer.step()  # This adjusts the model’s weights based on the gradients

    if epoch % 100 == 0:  # Print every 100 epochs
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Final prediction after training
with torch.no_grad():  # We don't need gradients during testing
    new_input = torch.tensor([[13.0]])
    prediction = model(new_input)
    print(f"Prediction for input 5: {prediction.item()}")
