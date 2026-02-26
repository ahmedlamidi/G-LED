import numpy as np
import matplotlib.pyplot as plt

# Read loss values from file
loss_file = "validation/loss.txt"
losses = np.loadtxt(loss_file)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(losses, linewidth=2, color='blue', marker='o', markersize=3)

# Label the axes
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Over Epochs', fontsize=14)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Tight layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig('loss_plot.png', dpi=150)
print(f"Plot saved as 'loss_plot.png'")

# Show the plot
plt.show()
