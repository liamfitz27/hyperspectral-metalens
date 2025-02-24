#%%
import torch

def fftconvolve(input, kernel, mode="full", pad=0, axes=None):
    """
    Perform convolution of two tensors using the FFT (Fast Fourier Transform).

    Args:
        input (torch.Tensor): The input tensor to be convolved.
        kernel (torch.Tensor): The kernel tensor to convolve with the input.
        mode (str, optional): The mode of convolution. Options are "full" (default) or "same".

        pad (int, optional): Additional padding to avoid circularity artifacts. Default is 0.
        axes (tuple of int, optional): The axes along which to perform the convolution. Default is None, which means all axes.

    Returns:
        torch.Tensor: The result of the convolution.
    """
    # Default axes: all axes if input and kernel have the same size
    if axes is None:
        axes = tuple(range(input.ndim))
    # Get input and kernel shapes along the specified axes
    input_shape = [input.shape[i] for i in axes]
    kernel_shape = [kernel.shape[i] for i in axes]
    # Compute the size of the output for full linear convolution
    full_shape = [input_shape[i] + kernel_shape[i] - 1 - 1*(kernel_shape[i]%2==0) for i in range(len(axes))]
    # Add extra padding to avoid circularity artifacts
    padded_shape = [s + 2 * pad for s in full_shape]
    # Calculate padding for input and kernel (only along specified axes)
    input_pad_width = [0] * (2 * input.ndim)  # Initialize padding for all axes
    kernel_pad_width = [0] * (2 * kernel.ndim)  # Initialize padding for all axes
    for i, axis in enumerate(axes):
        # Padding for input
        input_pad = padded_shape[i] - input_shape[i]
        input_pad_width[-2 * axis - 1] = input_pad // 2  # Left padding
        input_pad_width[-2 * axis - 2] = input_pad - input_pad // 2  # Right padding
        
        # Padding for kernel
        kernel_pad = padded_shape[i] - kernel_shape[i]
        kernel_pad_width[-2 * axis - 1] = kernel_pad // 2  # Left padding
        kernel_pad_width[-2 * axis + -2] = kernel_pad - kernel_pad // 2  # Right padding
    # Pad input and kernel (only along specified axes)
    input_padded = torch.nn.functional.pad(input, input_pad_width, mode='constant')
    kernel_padded =torch.nn.functional.pad(kernel, kernel_pad_width, mode='constant')
    # Convolution Theorem + FFT
    input_fft = torch.fft.fftn(input_padded, dim=axes)
    kernel_fft = torch.fft.fftn(kernel_padded, dim=axes)
    output_freq = input_fft * kernel_fft
    output = torch.fft.fftshift(torch.fft.ifftn(output_freq, dim=axes).real, dim=axes)
    # Remove extra padding
    if pad > 0:
        slices = [slice(pad, -pad) if i in axes else slice(None) for i in range(output.ndim)]
        output = output[tuple(slices)]
    # Crop to the same size as the input along the specified axes for mode="same"
    if mode == "same":
        crop_slices = [
            slice((full_shape[i] - input_shape[i]) // 2, (full_shape[i] - input_shape[i]) // 2 + input_shape[i])
            for i in range(len(axes))
        ]
        output = output[tuple(crop_slices)]
    return output


#%%
if __name__ == "__main__":
    import scipy.signal
    import matplotlib.pyplot as plt

    # Generate test input and kernel
    input = torch.randn(5, 5)  # 2D input
    kernel = torch.randn(3, 3)  # 2D kernels
    input_np = input.numpy()
    kernel_np = kernel.numpy()

    # Perform convolution using PyTorch
    output_full_torch = fftconvolve(input, kernel, mode="full")
    output_same_torch = fftconvolve(input, kernel, mode="same")
    # Perform convolution using scipy.signal.fftconvolve
    output_full_scipy = scipy.signal.fftconvolve(input_np, kernel_np, mode="full")
    output_same_scipy = scipy.signal.fftconvolve(input_np, kernel_np, mode="same")
    # Perform convolution using scipy.signal.convolve2d (linear convolution)
    output_full_linear = scipy.signal.convolve2d(input_np, kernel_np, mode="full")
    output_same_linear = scipy.signal.convolve2d(input_np, kernel_np, mode="same")

    def plot_results(input, kernel, output_torch, output_scipy, output_linear, title):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 5, 1)
        plt.imshow(input)
        plt.title("Input")
        plt.subplot(1, 5, 2)
        plt.imshow(kernel)
        plt.title("Kernel")
        plt.subplot(1, 5, 3)
        plt.imshow(output_torch)
        plt.title(f"TorchFFT {title}")
        plt.subplot(1, 5, 4)
        plt.imshow(output_scipy)
        plt.title(f"ScipyFFT {title}")
        plt.subplot(1, 5, 5)
        plt.imshow(output_linear)
        plt.title(f"Scipy {title}")

    # Test for mode="full"
    print("Testing mode='full'")
    plot_results(input_np, kernel_np, output_full_torch.numpy(), output_full_scipy, output_full_linear, "Full Convolution")

    # Test for mode="same"
    print("Testing mode='same'")
    plot_results(input_np, kernel_np, output_same_torch.numpy(), output_same_scipy, output_same_linear, "Same Convolution")