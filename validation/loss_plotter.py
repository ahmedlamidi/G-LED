import matplotlib.pyplot as plt

def plot_loss(input_file="validation/loss.txt", output_file="loss_plot.png"):
    epochs = []
    losses = []
    
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                epochs.append(int(parts[0]))
                losses.append(float(parts[1]))
            else:
                # If only loss values, use line number as epoch
                losses.append(float(parts[0]))
                epochs.append(len(losses))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs 720/816 model')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Loss plot saved to {output_file}")

if __name__ == "__main__":
    plot_loss()
