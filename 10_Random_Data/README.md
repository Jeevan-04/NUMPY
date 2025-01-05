# NumPy Random Data

## Table of Contents
1. [Introduction](#introduction)
2. [Random Sampling](#random-sampling)
3. [Distributions](#distributions)
4. [Shuffle and Permutation](#shuffle-and-permutation)
5. [Random Integers](#random-integers)
6. [Random State](#random-state)

## Introduction
This document provides examples and descriptions of various functions available in the NumPy random module for generating random data.

## Random Sampling
- `np.random.seed()`: Seed the random number generator.
- `np.random.rand()`: Generate random floats in the range [0.0, 1.0).
- `np.random.randn()`: Generate samples from the standard normal distribution.
- `np.random.randint()`: Generate random integers from low (inclusive) to high (exclusive).
- `np.random.random()`: Generate random floats in the range [0.0, 1.0).
- `np.random.random_sample()`: Generate random floats in the range [0.0, 1.0).
- `np.random.sample()`: Generate random floats in the range [0.0, 1.0).
- `np.random.choice()`: Randomly choose elements from an array.
- `np.random.bytes()`: Generate random bytes.

## Distributions
- `np.random.uniform()`: Generate samples from a uniform distribution.
- `np.random.normal()`: Generate samples from a normal (Gaussian) distribution.
- `np.random.binomial()`: Generate samples from a binomial distribution.
- `np.random.poisson()`: Generate samples from a Poisson distribution.
- `np.random.exponential()`: Generate samples from an exponential distribution.
- `np.random.chisquare()`: Generate samples from a chi-square distribution.
- `np.random.gamma()`: Generate samples from a gamma distribution.
- `np.random.beta()`: Generate samples from a beta distribution.
- `np.random.geometric()`: Generate samples from a geometric distribution.
- `np.random.hypergeometric()`: Generate samples from a hypergeometric distribution.
- `np.random.multinomial()`: Generate samples from a multinomial distribution.
- `np.random.negative_binomial()`: Generate samples from a negative binomial distribution.
- `np.random.pareto()`: Generate samples from a Pareto distribution.
- `np.random.weibull()`: Generate samples from a Weibull distribution.
- `np.random.logistic()`: Generate samples from a logistic distribution.
- `np.random.gumbel()`: Generate samples from a Gumbel distribution.
- `np.random.laplace()`: Generate samples from a Laplace distribution.
- `np.random.rayleigh()`: Generate samples from a Rayleigh distribution.
- `np.random.wald()`: Generate samples from a Wald distribution.
- `np.random.triangular()`: Generate samples from a triangular distribution.
- `np.random.f()`: Generate samples from an F distribution.

## Shuffle and Permutation
- `np.random.shuffle()`: Shuffle the array in-place.
- `np.random.permutation()`: Return a permuted array.

## Random Integers
- `np.random.randint()`: Generate random integers from low (inclusive) to high (exclusive).

## Random State
- `np.random.RandomState()`: Container for the Mersenne Twister pseudo-random number generator.
