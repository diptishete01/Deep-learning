import tensorflow as tf
import subprocess

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU Devices: {gpus}")

# Check detailed GPU info only if GPUs are detected by TensorFlow
if gpus:
    print("\nRunning nvidia-smi for detailed GPU information:")
    try:
        # Use subprocess to run the command and capture output/errors
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print(result.stdout)
    except FileNotFoundError:
        print("nvidia-smi command not found. This usually means NVIDIA drivers are not installed or configured.")
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}\nStderr: {e.stderr}")
else:
    print("\nNo GPU devices detected by TensorFlow, skipping nvidia-smi command.")
    print("To enable a GPU, go to Runtime > Change runtime type and select 'GPU' as the hardware accelerator.")
