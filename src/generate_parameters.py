import pandas as pd
from itertools import product

# Define the parameter values
optimizer_types = ['Adam', 'Adamax', 'Adagrad', 'RMSprop', 'SGD (Stochastic Gradient Descent)', 'FTRL (Follow-The-Regularized-Leader)']
learning_rates = [0.01, 0.001, 0.0001, 0.00003, 0.00001, 0.000001]
epochs = [5, 10, 15, 20, 25, 30]
batch_sizes = [8, 16, 32, 64, 128, 256]

# Generate all combinations of parameters
combinations = list(product(optimizer_types, learning_rates, epochs, batch_sizes))

# Create a DataFrame to store the combinations
df = pd.DataFrame(combinations, columns=['Optimizer Type', 'Learning Rate', 'Epochs', 'Batch Size'])

# Export the DataFrame to a CSV file
df.to_csv('testing/new_hyperparameters_tweaks.ods', index=False)
