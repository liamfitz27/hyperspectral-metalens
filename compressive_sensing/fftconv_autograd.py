#%%
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad

# NEED TO ADD PADDIG
def fft_convolve2d(x, k, mode="same", corr=False):
    if mode == "full":
        # Full shape
        s1, s2 = x.shape[0] + k.shape[0] - 2, x.shape[1] + k.shape[1] - 2
        # Zero pad x and k
        padx = ((s1 - x.shape[0])//2, (s2 - x.shape[1])//2)
        padk = ((s1 - k.shape[0])//2, (s2 - k.shape[1])//2)
        x_padded = jnp.pad(x, ((padx[0], padx[0]), (padx[1], padx[1])))
        k_padded = jnp.pad(k, ((padk[0], padk[0]), (padk[1], padk[1])))
        # FFT on by convolution theorem
        X = jnp.fft.fft2(x_padded)
        K = jnp.fft.fft2(k_padded)
        if corr:
            K = jnp.conj(K)
        conv_result = jnp.abs(jnp.fft.ifft2(X * K))
    elif mode == "same":
        if x.shape[0] > k.shape[0]:
            pad = (x.shape[0] - k.shape[0])//2
            pad = ((pad, pad), (0, 0))
            k = jnp.pad(k, pad)
        if x.shape[1] > k.shape[1]:
            pad = (x.shape[1] - k.shape[1])//2
            pad = ((0, 0), (pad, pad))
            k = jnp.pad(k, pad)
        X = jnp.fft.fft2(x)
        K = jnp.fft.fft2(k)
        if corr:
            K = jnp.conj(K)
        conv_result = jnp.fft.ifft2(X * K).real
    return jnp.fft.fftshift(conv_result)

#%%
if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    # Create a differentiable 2D variable
    x = jnp.array(np.random.randn(10, 10), dtype=jnp.float32)
    k = jnp.array(np.random.randn(10, 10), dtype=jnp.float32)
    conv_result = fft_convolve2d(x, k)
    plt.imshow(np.array(jnp.abs(conv_result)))
    # Define a differentiable loss function
    def loss_fn(x):
        conv_result = fft_convolve2d(x, k)
        return jnp.sum(jnp.abs(conv_result)**2)
    # Compute gradients
    grad_fn = grad(loss_fn)
    x_grad = grad_fn(x)
    print("Gradient of x:")
    print(x_grad)
    plt.figure()
    plt.imshow(np.array(x_grad))
