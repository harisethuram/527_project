import subprocess

if __name__ == "__main__":
    models = ["cnn", "clip", "transformer"]
    batch_sizes = [16, 32, 64, 128]
    epochs = [10, 20, 50, 100]
    
    for model in models:
        for batch_size in batch_sizes:
            for epoch in epochs:
                output_dir = f"models/{model}/batch_size_{batch_size}/epochs_{epoch}"
                print(f"Training {model} with batch size {batch_size} and epochs {epoch}")
                subprocess.run(["sbatch", "run/train.sh", model, str(batch_size), str(epoch), output_dir])
                