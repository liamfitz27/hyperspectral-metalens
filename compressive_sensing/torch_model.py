#%%
import torch

# Image model functions #

def rs_kernel(x, y, z, wvl):
    """
    Calculate the Rayleigh-Sommerfeld diffraction kernel.

    Parameters:
    x (torch.Tensor): The x-coordinate tensor.
    y (torch.Tensor): The y-coordinate tensor.
    z (torch.Tensor): The z-coordinate tensor.
    wvl (float): The wavelength of the light.

    Returns:
    torch.Tensor: The computed Rayleigh-Sommerfeld diffraction kernel.
    """
    k = 2 * torch.pi / wvl
    r = torch.sqrt(x**2 + y**2 + z**2)
    kernel = (1 / (2 * torch.pi)) * (1 / r - 1j * k) * (z / r) * torch.exp(1j * k * r) / r
    return kernel


def psf_fftconv(x, y, z, wvl, c, pad=0):
    """
    Computes the PSF using 2D FFT-based convolution along the x and y axes.

    Args:
        x (torch.Tensor): 1D tensor of x coordinates.
        y (torch.Tensor): 1D tensor of y coordinates.
        z (torch.Tensor): 1D tensor of z coordinates (propagation distances).
        wvl (torch.Tensor): 1D tensor of wavelengths.
        c (torch.Tensor): Input field of shape (len(x), len(y), len(wvl)).
        pad (int): Padding size for FFT convolution.

    Returns:
        torch.Tensor: PSF of shape (len(x), len(y), len(z), len(wvl)).
    """
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    # Reshape xx, yy, z, wvl, c for broadcasting
    xx = xx.unsqueeze(2).unsqueeze(3)  # Shape: (len(x), len(y), 1, 1)
    yy = yy.unsqueeze(2).unsqueeze(3)  # Shape: (len(x), len(y), 1, 1)
    z = z.view(1, 1, -1, 1)  # Shape: (1, 1, len(z), 1)
    wvl = wvl.view(1, 1, 1, -1)  # Shape: (1, 1, 1, len(wvl))
    c = c.unsqueeze(2)  # Shape: (len(x), len(y), 1, len(wvl))

    # Rayleigh-Sommerfeld diffraction kernel
    kernel = rs_kernel(xx, yy, z, wvl)  # Shape: (len(x), len(y), len(z), len(wvl))

    # Perform 2D FFT-based convolution along x and y axes
    psf = fftconvolve(c, kernel, complex=True, mode="same", pad=pad, axes=(0, 1))
    return psf


def focus_phase(x, y, focl, wvl_cen, cen=(0, 0)):
    """
    Calculate the focusing phase for a given set of (x,y) coordinates and wavelengths.

    Args:
        x (torch.Tensor): 1D tensor of x-coordinates.
        y (torch.Tensor): 1D tensor of y-coordinates.
        focl (float): Focal length.
        wvl_cen (torch.Tensor): 1D tensor of central wavelengths.
        cen (tuple, optional): Tuple representing the center coordinates (default is (0, 0)).

    Returns:
        torch.Tensor: 3D tensor of complex values representing the focusing phase.
    """
    # Init tensors
    c_foc = torch.empty((len(x), len(y), len(wvl_cen)), dtype=torch.complex64)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    # Calculate hyperbolic phase shift for focusing for each wavelength
    for l in range(len(wvl_cen)):
        phi_foc = 2 * torch.pi / wvl_cen[l] * (focl - torch.sqrt(focl**2 + (xx - cen[0])**2 + (yy - cen[1])**2))
        # Compute complex exponential of phase shift
        c_foc[:, :, l] = torch.exp(1j * phi_foc)
    return c_foc


def refract_index_fill(n0, lx, h, wvl):
    """
    Compute the phase shift due to relative refractive index variation with area fill factor (constant height).

    Args:
        n0 (float): Refractive index of the medium.
        lx (torch.Tensor): 1D tensor of spatial coordinates.
        h (float): Constant height value.
        wvl (torch.Tensor): 1D tensor of wavelengths.

    Returns:
        torch.Tensor: Phase shift tensor of shape (len(lx), len(lx), len(wvl)).
    """
    # Compute the area fill factor (Afrac = lx^2)
    Afrac = lx**2  # Shape: (len(lx),)

    # Compute the phase term (2 * pi * (1 - n0) * h / wvl)
    phase_term = 2 * torch.pi * (1 - n0) * h / wvl  # Shape: (len(wvl),)

    # Perform the multiplication (star) with broadcasting
    phase = Afrac.unsqueeze(-1) * phase_term.unsqueeze(0)  # Shape: (len(lx), len(wvl))

    return phase


# Compressive sensing functions #

def decode_dict_fftconv(psf_spect, sparse_dict, img_shape):
    """
    Decode a sparse dictionary using FFT-based convolution with a PSF spectrum.

    Args:
        psf_spect (torch.Tensor): PSF spectrum of shape (nx, ny, nwvl).
        sparse_dict (torch.Tensor): Sparse dictionary of shape (nx_full * ny_full * nwvl, s1).
        img_shape (tuple): Shape of the full image (nx_full, ny_full, nwvl).

    Returns:
        torch.Tensor: Decoded dictionary of shape (nx_full * ny_full, s1).
    """
    # Reshape sparse_dict to (nx_full, ny_full, nwvl, s1)
    s0 = int(sparse_dict.shape[0] / img_shape[-1])  # Full spatial size (nx_full, ny_full)
    s1 = sparse_dict.shape[1]  # Number of basis functions
    basis_set = sparse_dict.reshape(*img_shape, s1)  # Shape: (nx_full, ny_full, nwvl, s1)

    # Repeat psf_spect along the basis dimension
    psf_spect = psf_spect.unsqueeze(-1).repeat(1, 1, 1, s1)  # Shape: (nx, ny, nwvl, s1)

    # Perform convolution over spatial axes (0, 1)
    basis_psf_conv = fftconvolve(basis_set, psf_spect, mode="same", axes=(0, 1))  # Shape: (nx_full, ny_full, nwvl, s1)
    
    # Sum along the wavelength dimension
    basis_psf_conv = torch.sum(basis_psf_conv, dim=2)  # Shape: (nx_full, ny_full, s1)

    # Reshape to (nx_full * ny_full, s1)
    decode_dict = basis_psf_conv.reshape(s0, s1)

    return decode_dict


def mutual_coherence(A):
    """
    Compute the mutual coherence of a matrix.

    Args:
        A (torch.Tensor): Input matrix of shape (n, m).

    Returns:
        float: Mutual coherence value.
    """
    # Normalize the rows of A
    norm = torch.sum(torch.abs(A), dim=0, keepdim=True)
    A_norm = A / norm
    A_norm = torch.where(norm == 0, 0, A_norm)

    # Compute the mutual coherence matrix
    mc_mat = A_norm.T @ A_norm  # Shape: (n, n)
 
    # Extract unique pairs (upper triangular part, excluding the diagonal)
    unique_pairs = torch.triu(mc_mat, diagonal=1)

    # A_norm = torch.sum(torch.abs(A), axis=0, keepdims=True)
    # A_norm = torch.where(A_norm == 0, 0, A / A_norm)
    # term1 = torch.sum(torch.sum(A_norm, axis=1)**2)
    # term2 = torch.sum(A_norm**2)

    # Sum the unique pairs to compute mutual coherence
    return torch.sum(unique_pairs) #term1 - term2



# Torch FFT Convolve #

def fftconvolve(input, kernel, complex=False, mode="full", pad=0, axes=None):
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
    if complex:
        output = torch.fft.fftshift(torch.fft.ifftn(output_freq, dim=axes), dim=axes)
    else:
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

# Torch FFT Convolve #



#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    import numpy as np
    # TEST PSF FFT CONVOLUTION
    # Parameters
    lx = 50  # Size of the grid in x and y (from -50 to 50)
    focl = 200  # Focal length
    nx = 50
    wvl_rgb = torch.tensor([0.650, 0.510, 0.475])  # Wavelengths for red, green, and blue (in micrometers)
    x = torch.linspace(-lx/2, lx/2, nx)  # x coordinates
    y = torch.linspace(-lx/2, lx/2, nx)  # y coordinates
    z = torch.tensor([focl])  # Propagation distance (focal plane)

    # Generate the focusing phase
    c = focus_phase(x, y, focl, wvl_rgb)

    # Compute the PSF
    psf = psf_fftconv(x, y, z, wvl_rgb, c, pad=20)
    I_focl = torch.abs(psf[:, :, 0, :])**2

    from psf_fft_autograd import focus_phase as focus_phase2
    from psf_fft_autograd import psf_fftconv as psf_fftconv2
    import jax.numpy as jnp
    wvl_rgb2 = jnp.array([0.650, 0.510, 0.475])  # Wavelengths for red, green, and blue (in micrometers)
    x = jnp.linspace(-lx/2, lx/2, nx)  # x coordinates
    y = jnp.linspace(-lx/2, lx/2, nx)  # y coordinates
    z = jnp.array([focl])
    c_foc = focus_phase2(x, x, focl, wvl_rgb2)
    psf_spect = psf_fftconv2(x, x, [focl], wvl_rgb2, c_foc, pad=20)
    I_focl2 = jnp.abs(psf_spect[:, :, 0, :])**2
    

    # Plot |PSF|² at z = focl for each wavelength
    fig, ax = plt.subplots(2,3, dpi=200)
    titles = ["Red (650 nm)", "Green (510 nm)", "Blue (475 nm)"]
    for l in range(len(wvl_rgb)):
        ax[0,l].imshow(I_focl[:,:,l], extent=[-lx/2, lx/2, -lx/2, lx/2] , vmin=0, vmax=I_focl.max())
        ax[1,l].imshow(I_focl2[:,:,l], extent=[-lx/2, lx/2, -lx/2, lx/2], vmin=0, vmax=I_focl2.max())
        ax[0,l].set_title(titles[l])
        ax[1,l].set_title(titles[l])
        plt.xlabel("x (µm)")
        plt.ylabel("y (µm)")
    plt.tight_layout()
    plt.show()

     #%%
    # TEST REFRACT INDEX FILL
    n0 = 4
    h = 0.7
    lx = torch.angle(c[:,:,0])%(2*torch.pi)
    lx = lx / lx.max()
    c = refract_index_fill(n0, lx, h, wvl_rgb)

    from psf_fft_autograd import refract_index_fill as refract_index_fill2
    lx = jnp.angle(c_foc[:,:,0])%(2*jnp.pi)
    lx = lx / lx.max()
    c2 = refract_index_fill2(n0, lx, h, wvl_rgb2)

    fig, ax = plt.subplots(2,3,dpi=150)
    for i in range(3):
        ax[0,i].imshow(c[:,:,i])
        ax[1,i].imshow(c2[:,:,i])
    fig.tight_layout()

    #%%
    # TEST DECODE DICT FFT CONVOLVE
    from decoding_dict import DWT_dict_full
    
    wavelet_dict = DWT_dict_full(nx, 3, nx, type="haar")
    wavelet_dict = torch.tensor(wavelet_dict, dtype=torch.float32)
    decode_dict = decode_dict_fftconv(I_focl, wavelet_dict, [2*nx-2, 2*nx-2, 3])
    plt.figure()
    plt.imshow(decode_dict)

    from decoding_dict_autograd import decode_dict_fftconv as decode_dict_fftconv2

    wavelet_dict = jnp.array(DWT_dict_full(nx, 3, nx, type="haar"))
    decode_dict2 = decode_dict_fftconv2(I_focl2, wavelet_dict, [2*nx-2, 2*nx-2, 3])
    plt.figure()
    plt.imshow(decode_dict2)

    rel_diff = 2*(np.abs((np.array(decode_dict) - np.array(decode_dict2))/(np.array(decode_dict) + np.array(decode_dict2))))
    rel_diff = np.where(np.array(decode_dict) + np.array(decode_dict2) == 0, 0, rel_diff)
    rel_diff_pc = 100 * rel_diff
    plt.figure()
    plt.imshow(rel_diff_pc)
    plt.colorbar()

    #%%
    # TEST MUTUAL COHERENCE
    mc = mutual_coherence(decode_dict)
    print(f"Mutual coherence: {mc}")

    def mutual_coherence2(phi):
        phi_norm = jnp.sum(jnp.abs(phi), axis=0, keepdims=True)
        phi_norm = jnp.where(phi_norm == 0, 0, phi / phi_norm)
        mc_mat = phi_norm.T @ phi_norm
        unique_pairs = jnp.triu(mc_mat, k=1)
        sum = jnp.sum(unique_pairs)
        return sum
    
    decode_dict2 = jnp.array(np.array(decode_dict))
    mc2 = mutual_coherence2(decode_dict2)
    print(f"Mutual coherence2: {mc2}")