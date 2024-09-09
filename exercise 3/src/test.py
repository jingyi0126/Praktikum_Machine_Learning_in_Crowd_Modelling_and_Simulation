import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
cuda_available = torch.cuda.is_available()
if cuda_available:
    # Print the number of GPUs available
    print("Number of GPUs:", torch.cuda.device_count())

    # Get the name of the current GPU
    print(
        "Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device())
    )
else:
    print("CUDA is not available. Check your installation and GPU drivers.")

if cuda_available:
    # Create a tensor and move it to GPU
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print("Tensor on GPU:", x)
else:
    print("CUDA is not available. Check your installation and GPU drivers.")

if cuda_available:
    y = x * 2  # Perform a simple computation on the GPU
    print("Computed on GPU:", y)
    print(
        "Computed on GPU and moved to CPU:", y.cpu()
    )  # Move the result back to CPU to print it
else:
    print("CUDA is not available. Check your installation and GPU drivers.")
