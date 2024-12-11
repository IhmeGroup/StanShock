import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import beta as beta_func
from scipy.stats import beta, gaussian_kde
from scipy.integrate import quad, simps

def split_array(arr, n_chunks):
    """Split an array into n_chunks"""
    chunk_size = len(arr) // n_chunks
    return [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]

class BetaKDE:
    def __init__(self, data, weights=None, bandwidth=None):
        self.data = np.array(data)
        self.n = len(self.data)
        
        # Handle weights
        if weights is None:
            self.weights = np.ones(self.n)
        else:
            self.weights = np.array(weights)
            if len(self.weights) != self.n:
                raise ValueError("The number of weights must match the number of data points")
        self.weights = self.weights / np.sum(self.weights)
        
        # If bandwidth is not provided, use a rule of thumb
        if bandwidth is None:
            self.bandwidth = self.n**(-2/5) # Chen 1999
            # self.bandwidth = self.n**(-1)
        else:
            self.bandwidth = bandwidth
        
        # Precompute values for faster evaluation
        self.a_params = self.data / self.bandwidth + 1
        self.b_params = (1 - self.data) / self.bandwidth + 1

    def __call__(self, x):
        """Evaluate the weighted kernel density estimate at point(s) x"""
        x = np.atleast_1d(x)
        x = x[:, np.newaxis]  # Add a new axis for broadcasting
        kernel_values = beta.pdf(x, self.a_params, self.b_params)
        return np.sum(kernel_values * self.weights, axis=1)

class MonteCarloSampler2D:
    def __init__(self, ranges, func, grad_func, grid_dims, num_samples_per_iter=100, max_iters=20,
                 alpha=0.5, epsilon=1e-6, var_tol=1e-4):
        self.ranges = ranges  # List of (min, max) for each dimension
        self.num_samples_per_iter = num_samples_per_iter
        self.max_iters = max_iters
        self.alpha = alpha
        self.epsilon = epsilon
        self.var_tol = var_tol
        self.func = func
        self.grad_func = grad_func
        
        # Create grid
        self.x_edges = np.linspace(*ranges[0], grid_dims[0]+1)
        self.y_edges = np.linspace(*ranges[1], grid_dims[1]+1)
        self.x_centers = (self.x_edges[1:] + self.x_edges[:-1]) / 2
        self.y_centers = (self.y_edges[1:] + self.y_edges[:-1]) / 2
        self.x_centers_grid, self.y_centers_grid = np.meshgrid(self.x_centers, self.y_centers)

        # Interpolators for inverse transform sampling
        self.P_x_cumsum = None
        self.x_intp = None
        self.y_intps = None

    def compute_sampling_density(self, *coords):
        """Compute the sampling density P(x, y) for a chunk of points"""
        if self.alpha == 0.0:
            return np.ones(coords[0].shape)

        grad = self.grad_func(*coords)
        grad_mag = np.linalg.norm(grad, axis=0)
        density = np.maximum(self.epsilon, grad_mag**self.alpha)
        density /= density.sum()
        return density
    
    def inverse_transform_sampling(self, num_samples):
        """Generate samples using 2D inverse transform sampling"""
        if self.alpha == 0.0:
            # Case where the density is uniform
            x_samples = np.random.uniform(*self.ranges[0], num_samples)
            y_samples = np.random.uniform(*self.ranges[1], num_samples)
            return x_samples, y_samples

        if self.P_x_cumsum is None:
            # Compute the sampling density map based on the specified grid
            P_xy = self.compute_sampling_density(self.x_centers_grid, self.y_centers_grid)
            P_x = P_xy.sum(axis=0)
            self.P_x_cumsum = np.cumsum(P_x)
            self.P_x_cumsum = np.insert(self.P_x_cumsum, 0, 0)
            self.x_intp = interp1d(self.P_x_cumsum, self.x_edges)
            self.y_intps = [interp1d(np.cumsum(np.insert(P_xy[:, i] / P_x[i], 0, 0)), self.y_edges) 
                            for i in range(len(self.x_edges)-1)]
            
        x_samples = np.zeros(num_samples)
        y_samples = np.zeros(num_samples)
        uniform_samples = np.random.random((num_samples, 2))

        for i, (ux, uy) in enumerate(uniform_samples):
            x_idx = np.searchsorted(self.P_x_cumsum, ux) - 1
            x_samples[i] = self.x_intp(ux)
            y_samples[i] = self.y_intps[x_idx](uy)

        return x_samples, y_samples

    def estimate_pdf(self, samples, weights):
        """Estimate the PDF of phi"""
        # Use Beta distribution KDE
        return BetaKDE(samples, weights, bandwidth=0.001)

        # Use Gaussian KDE
        # return gaussian_kde(samples, weights=weights)

        # Use histogram
        n_bins = int(np.sqrt(len(samples)))
        bin_edges = np.linspace(0, 1, n_bins+1)
        hist, bin_edges = np.histogram(samples, bins=bin_edges, weights=weights, density=True)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        return interp1d(bin_centers, hist, kind='nearest', fill_value="extrapolate")

    def compute_scalar_pdf(self):
        """Run the sampling simulation"""
        x_samples = np.array([])
        y_samples = np.array([])
        phi_samples = np.array([])
        weights = np.array([])
        variance_old = 0
        diff_variance_norm = np.inf
        iter = 0

        while diff_variance_norm > self.var_tol and iter < self.max_iters:
            # Generate samples
            x_samples_new, y_samples_new = self.inverse_transform_sampling(self.num_samples_per_iter)

            # Compute phi values and weights for the samples
            phi_samples_new = self.func(x_samples_new, y_samples_new)

            # Early stopping if all samples are close to 0
            min_maxval = 1e-6
            if phi_samples_new.max() < min_maxval:
                # Case where all samples are close to 0. Assume delta distribution at 0.
                phi_pdf = lambda x: np.less(x, min_maxval) / min_maxval
                return phi_pdf, np.array([]), np.array([]), np.array([]), np.array([])

            # Compute weights (inverse of sampling density)
            weights_new = 1 / self.compute_sampling_density(x_samples_new, y_samples_new)
            weights_new /= weights_new.sum()

            # Append new samples
            x_samples = np.concatenate([x_samples, x_samples_new])
            y_samples = np.concatenate([y_samples, y_samples_new])
            phi_samples = np.concatenate([phi_samples, phi_samples_new])
            weights = np.concatenate([weights, weights_new])
            weights /= weights.sum()

            # Estimate p(phi)
            phi_pdf = self.estimate_pdf(phi_samples, weights)

            # Compute the mean and variance of the distribution
            mean = quad(lambda x: x * phi_pdf(x), 0, 1)[0]
            variance = quad(lambda x: (x - mean)**2 * phi_pdf(x), 0, 1)[0]

            # Compute the mean and variance of the samples
            # mean = np.sum(phi_samples * weights) / np.sum(weights)
            # variance = np.sum(weights * (phi_samples - mean)**2)

            diff_variance_norm = np.abs(variance - variance_old) / variance_old
            variance_old = variance
            
            print(f"iter: {iter}, " +
                  f"n_samples: {len(phi_samples)}, " +
                  f"mean: {mean}, " +
                  f"variance: {variance}, " +
                  f"diff_variance_norm: {diff_variance_norm}")

            iter += 1

        return phi_pdf, x_samples, y_samples, phi_samples, weights

    def plot_results(self, func, x_samples, y_samples, phi_samples, weights, phi_pdf):
        """Plot the results"""
        max_samples_plot = 10000
        if len(x_samples) > max_samples_plot:
            idx = np.random.choice(len(x_samples), max_samples_plot, replace=False)
            x_samples, y_samples, phi_samples, weights = x_samples[idx], y_samples[idx], phi_samples[idx], weights[idx]

        plt.figure(figsize=(10, 5))

        # Plot the function
        plt.subplot(121)
        x = np.linspace(*self.ranges[0], 100)
        y = np.linspace(*self.ranges[1], 100)
        x_grid, y_grid = np.meshgrid(x, y)
        z = np.array([[func(i, j) for i in x] for j in y])
        plt.contourf(x_grid, y_grid, z)
        plt.scatter(x_samples, y_samples, c='white', s=1)
        plt.colorbar(label='φ')
        plt.title('Function')

        # Plot p(φ)
        plt.subplot(122)
        phi_range = np.linspace(0, 1, 1000)
        plt.plot(phi_range, phi_pdf(phi_range))
        plt.title('p(φ)')
        plt.xlabel('φ')

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Define parameters
    x_range, y_range = [-2, 2], [-2, 2]
    ranges = [x_range, y_range]
    centers = [(0, 0), (1, 1), (-1, -1)]
    sigma = 0.5

    # Define Gaussian mixture and gradient functions
    def gaussian_mixture(x, y):
        result = 0
        for cx, cy in centers:
            result += np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        return result

    def gradient_magnitude(x, y):
        dx, dy = 0, 0
        for cx, cy in centers:
            factor = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
            dx += -(x - cx) / sigma**2 * factor
            dy += -(y - cy) / sigma**2 * factor
        return np.sqrt(dx**2 + dy**2)

    # Run simulation
    sampler = MonteCarloSampler2D(ranges, gaussian_mixture, gradient_magnitude)
    phi_pdf, x_samples, y_samples, phi_samples, weights = sampler.compute_scalar_pdf()

    # Plot results
    sampler.plot_results(gaussian_mixture, x_samples, y_samples, phi_samples, weights, phi_pdf)
