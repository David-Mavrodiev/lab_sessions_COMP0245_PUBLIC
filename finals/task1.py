import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

torch.set_num_threads(1)

# Constants
m = 1.0  # Mass (kg)
b = 10  # Friction coefficient
k_p = 50  # Proportional gain
k_d = 10   # Derivative gain
dt = 0.01  # Time step
num_samples = 1000  # Number of samples in dataset

# Generate synthetic data for trajectory tracking
t = np.linspace(0, 10, num_samples)
q_target = np.sin(t)
dot_q_target = np.cos(t)

# Initial conditions for training data generation
q = 0
dot_q = 0
X = []
Y = []

for i in range(num_samples):
    # PD control output
    tau = k_p * (q_target[i] - q) + k_d * (dot_q_target[i] - dot_q)
    # Ideal motor dynamics (variable mass for realism)
    #m_real = m * (1 + 0.1 * np.random.randn())  # Mass varies by +/-10%
    ddot_q_real = (tau - b * dot_q) / m
    
    # Calculate error
    ddot_q_ideal = (tau) / m
    ddot_q_error = ddot_q_ideal - ddot_q_real
    
    # Store data
    X.append([q, dot_q, q_target[i], dot_q_target[i]])
    Y.append(ddot_q_error)
    
    # Update state
    dot_q += ddot_q_real * dt
    q += dot_q * dt

# Convert data for PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32).to('cpu')
Y_tensor = torch.tensor(Y, dtype=torch.float32).to('cpu').view(-1, 1)

# Dataset and DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
#train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

class CorrectorMLP(nn.Module):
    def __init__(self, label, num_hidden_nodes, dataset, isDeepNetwork=False):
        super(CorrectorMLP, self).__init__()
        self.label = label

        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        if isDeepNetwork:
            self.layers = nn.Sequential(
                nn.Linear(4, num_hidden_nodes),
                nn.ReLU(),
                nn.Linear(num_hidden_nodes, num_hidden_nodes),  # Second hidden layer
                nn.ReLU(),
                nn.Linear(num_hidden_nodes, 1)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(4, num_hidden_nodes),
                nn.ReLU(),
                nn.Linear(num_hidden_nodes, 1)
            )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.00001)

    def forward(self, x):
        return self.layers(x)

    def train_model1(self, epochs = 1000):
        train_losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            train_losses.append(epoch_loss / len(self.train_loader))
            print(f'Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.6f}')

        return train_losses

    def train_model(self, epochs):
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            # Training
            self.train()
            epoch_train_loss = 0
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()

            train_losses.append(epoch_train_loss / len(self.train_loader))

            # Validation
            self.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for data, target in self.val_loader:
                    output = self(data)
                    loss = self.criterion(output, target)
                    epoch_val_loss += loss.item()

            val_losses.append(epoch_val_loss / len(self.val_loader))

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

        return train_losses, val_losses

    def evaluate_model(self, test_loader):
        self.eval()  # Set the model to evaluation mode
        test_loss = 0
        all_predictions = []

        with torch.no_grad():  # Disable gradient computation
            for data, target in test_loader:
                output = model(data)  # Forward pass
                loss = self.criterion(output, target)
                test_loss += loss.item()
                all_predictions.extend(output.squeeze().tolist())  # Collect predictions

        test_loss /= len(test_loader)  # Average test loss
        print(f'Test Loss: {test_loss:.6f}')
        return test_loss, all_predictions

def get_pd_results_without_correction():
    q_test = 0
    dot_q_test = 0
    q_real = []
    # Integration with only PD Control (without correction)
    for i in range(len(t)):
        tau = k_p * (q_target[i] - q_test) + k_d * (dot_q_target[i] - dot_q_test)
        ddot_q_real = (tau - b * dot_q_test) / m
        dot_q_test += ddot_q_real * dt
        q_test += dot_q_test * dt
        q_real.append(q_test)
    
    return q_real

def get_pd_results_with_correction(model):
    # Reset test conditions for PD + MLP Correction
    q_test = 0
    dot_q_test = 0
    q_real_corrected = []
    
    # Integration with PD + MLP Correction
    for i in range(len(t)):
        tau = k_p * (q_target[i] - q_test) + k_d * (dot_q_target[i] - dot_q_test)
        inputs = torch.tensor([q_test, dot_q_test, q_target[i], dot_q_target[i]], dtype=torch.float32).to('cpu')
        correction = model(inputs.unsqueeze(0)).item()
        ddot_q_corrected = (tau - b * dot_q_test + correction) / m
        dot_q_test += ddot_q_corrected * dt
        q_test += dot_q_test * dt
        q_real_corrected.append(q_test)

    return q_real_corrected

hidden_nodes_list = [32, 64, 96, 128]
train_losses_results = {}
validation_losses_results = {}

q_real_results = {}
q_real_corrected_results = {}

q_real_results[f'PD Only'] = get_pd_results_without_correction()

for nodes in hidden_nodes_list:
    model = CorrectorMLP(label=f'PD + MLP Correction, Hidden Nodes: {nodes}', num_hidden_nodes=nodes, dataset=dataset)
    train_losses, validation_losses = model.train_model(epochs = 1000)

    train_losses_results[f'Hidden Nodes: {nodes}'] = train_losses
    validation_losses_results[f'Hidden Nodes: {nodes}'] = validation_losses
    q_real_corrected_results[model.label] = get_pd_results_with_correction(model)

print(q_real_corrected_results)

plt.figure(figsize=(10, 6))
for nodes, losses in train_losses_results.items():
    plt.plot(losses, label=nodes)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss for Different Hidden Node Counts')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for nodes, losses in validation_losses_results.items():
    plt.plot(losses, label=nodes)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss for Different Hidden Node Counts')
plt.legend()
plt.show()

# Plot results for each configuration
plt.figure(figsize=(12, 6))
plt.plot(t, q_target, 'r-', label='Target')

# Plot PD Only and PD + MLP corrected for each hidden node count
for label, q_real in q_real_results.items():
    plt.plot(t, q_real, '--', label=label)

for label, q_real_corrected in q_real_corrected_results.items():
    plt.plot(t, q_real_corrected, ':', label=label)

plt.title('Trajectory Tracking with and without MLP Correction for Different Hidden Node Counts')
plt.xlabel('Time [s]')
plt.ylabel('Position')
plt.legend()
plt.show()

# Calculate the baseline error for PD Only
error_pd_only = np.array(q_target) - np.array(q_real_results['PD Only'])

# Calculate errors for PD + MLP Correction with each hidden node count
errors_pd_mlp_correction = {}
for label, q_real_corrected in q_real_corrected_results.items():
    errors_pd_mlp_correction[label] = np.array(q_target) - np.array(q_real_corrected)

# Calculate final error metrics (mean absolute error in this case)
final_errors = {}

# Baseline error for PD Only
final_errors['PD Only'] = np.mean(np.abs(error_pd_only))

# Errors for PD + MLP Correction for each hidden node configuration
for label, errors in errors_pd_mlp_correction.items():
    final_errors[label] = np.mean(np.abs(errors))

# Plotting the final error for each configuration
plt.figure(figsize=(10, 6))
plt.bar(final_errors.keys(), final_errors.values(), color=['blue', 'orange', 'green', 'red', 'purple'])
plt.xlabel('Model Type')
plt.ylabel('Mean Absolute Error')
plt.title('Final Error Comparison for PD Only and PD + MLP Correction with Different Hidden Node Counts')
plt.xticks(rotation=45)
plt.show()

# Training loss distribution
plt.figure(figsize=(10, 6))
plt.boxplot(train_losses_results.values(), labels=train_losses_results.keys())
plt.xlabel('Hidden Node Count')
plt.ylabel('Training Loss')
plt.title('Training Loss Distribution for Different Hidden Node Counts')
plt.show()

# Validation loss distribution
plt.figure(figsize=(10, 6))
plt.boxplot(validation_losses_results.values(), labels=validation_losses_results.keys())
plt.xlabel('Hidden Node Count')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Distribution for Different Hidden Node Counts')
plt.show()

plt.figure(figsize=(12, 6))
cumulative_error_pd_only = np.cumsum(np.abs(error_pd_only))
plt.plot(t, cumulative_error_pd_only, label='PD Only')

for label, errors in errors_pd_mlp_correction.items():
    cumulative_error_mlp = np.cumsum(np.abs(errors))
    plt.plot(t, cumulative_error_mlp, label=label)

plt.xlabel('Time [s]')
plt.ylabel('Cumulative Absolute Error')
plt.title('Cumulative Error Over Time for PD Only and PD + MLP Correction')
plt.legend()
plt.show()
