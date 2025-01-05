import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Random Sampling
print("Random Sampling:")
print(np.random.rand(3))  # Random floats in [0.0, 1.0)
print(np.random.randn(3))  # Samples from the standard normal distribution
print(np.random.randint(1, 10, 3))  # Random integers from low (inclusive) to high (exclusive)
print(np.random.random(3))  # Random floats in [0.0, 1.0)
print(np.random.random_sample(3))  # Random floats in [0.0, 1.0)
print(np.random.sample(3))  # Random floats in [0.0, 1.0)
print(np.random.choice([1, 2, 3, 4, 5], 3))  # Randomly choose elements from an array
print(np.random.bytes(5))  # Random bytes

# Distributions
print("\nDistributions:")
print(np.random.uniform(1.0, 2.0, 3))  # Uniform distribution
print(np.random.normal(0.0, 1.0, 3))  # Normal (Gaussian) distribution
print(np.random.binomial(10, 0.5, 3))  # Binomial distribution
print(np.random.poisson(5, 3))  # Poisson distribution
print(np.random.exponential(1.0, 3))  # Exponential distribution
print(np.random.chisquare(2, 3))  # Chi-Square distribution
print(np.random.gamma(2.0, 2.0, 3))  # Gamma distribution
print(np.random.beta(0.5, 0.5, 3))  # Beta distribution
print(np.random.geometric(0.5, 3))  # Geometric distribution
print(np.random.hypergeometric(10, 5, 3, 3))  # Hypergeometric distribution
print(np.random.multinomial(10, [0.2, 0.3, 0.5], 3))  # Multinomial distribution
print(np.random.negative_binomial(5, 0.5, 3))  # Negative Binomial distribution
print(np.random.pareto(2.0, 3))  # Pareto distribution
print(np.random.weibull(2.0, 3))  # Weibull distribution
print(np.random.logistic(0.0, 1.0, 3))  # Logistic distribution
print(np.random.gumbel(0.0, 1.0, 3))  # Gumbel distribution
print(np.random.laplace(0.0, 1.0, 3))  # Laplace distribution
print(np.random.rayleigh(1.0, 3))  # Rayleigh distribution
print(np.random.wald(1.0, 1.0, 3))  # Wald distribution
print(np.random.triangular(1.0, 2.0, 3.0, 3))  # Triangular distribution
print(np.random.f(2.0, 2.0, 3))  # F distribution

# Shuffle and Permutation
print("\nShuffle and Permutation:")
arr = np.array([1, 2, 3, 4, 5])
np.random.shuffle(arr)  # Shuffle the array in-place
print(arr)
print(np.random.permutation(arr))  # Return a permuted array

# Random Integers
print("\nRandom Integers:")
print(np.random.randint(1, 10, 3))  # Random integers from low (inclusive) to high (exclusive)

# Random State
print("\nRandom State:")
state = np.random.RandomState(42)
print(state.rand(3))  # Random floats in [0.0, 1.0)
