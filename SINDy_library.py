from itertools import combinations_with_replacement
import math
import torch
from scipy.special import binom
from scipy.integrate import odeint
import numpy as np


def sindy_library_pt(z, latent_dim, poly_order, include_sine=False):
    """
    Description: Builds the SINDy library for a first-order dynamical system.

    Args:
        z (torch.Tensor): Input tensor of shape (batch_size, latent_dim), representing latent states.
        latent_dim (int): Number of latent variables (dimensions).
        poly_order (int): Maximum degree of polynomial terms to include in the library.
        include_sine (bool): Whether to include sine terms in the library.

    Returns:
        torch.Tensor: A matrix (batch_size, library_size) where each column is a function of z.
    """

    # Initialize the library with a column of ones. The number of rows is equal to batch size.
    library = [torch.ones(z.size(0)).to(device="cuda")]

    # Prepare to loop over all variable combinations
    sample_list = range(latent_dim)

    for n in range(1, poly_order + 1):
        # Get all combinations (with replacement) of latent_dim variables of total degree n
        list_combinations = list(combinations_with_replacement(sample_list, n))

        for combination in list_combinations:
            # For each combination, compute the product of the corresponding columns in z
            # e.g., z[:, [0, 0]] -> z_0^2, z[:, [1, 2]] -> z_1 * z_2
            term = torch.prod(z[:, combination], dim=1)
            library.append(term.to(device="cuda"))  # Add to the library (on GPU)

    # Optionally add sine terms of each latent variable
    if include_sine:
        for i in range(latent_dim):
            library.append(
                torch.sin(z[:, i])
            )  # Automatically on correct device since z is

    # Stack all features column-wise into a single tensor of shape (batch_size, num_features)
    return torch.stack(library, dim=1).to(device="cuda")


def library_size(latent_dim, poly_order):
    f = lambda latent_dim, poly_order: math.comb(
        latent_dim + poly_order - 1, poly_order
    )
    total = 0
    for i in range(poly_order + 1):
        total += f(latent_dim, i)
    return total


def sindy_library_pt_order2(z, dz, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library for a second-order system.
    """
    library = [torch.ones(z.size(0)).to(device="cuda")]  # initialize library

    # concatenate z and dz
    z_combined = torch.cat([z, dz], dim=1)

    sample_list = range(2 * latent_dim)
    list_combinations = list()

    for n in range(1, poly_order + 1):
        list_combinations = list(combinations_with_replacement(sample_list, n))
        for combination in list_combinations:
            library.append(
                torch.prod(z_combined[:, combination], dim=1).to(device="cuda")
            )

    # add sine terms if included
    if include_sine:
        for i in range(2 * latent_dim):
            library.append(torch.sin(z_combined[:, i]))

    return torch.stack(library, dim=1).to(device="cuda")


def poly_add(z, library, order, latent_dim):
    for i in range(latent_dim):
        poly_product(z, library, order, i, latent_dim)


def poly_product(z, library, order, index, latent_dim, seen_combinations=None):
    if seen_combinations is None:
        seen_combinations = set()

    if order > 1:
        for j in range(index, latent_dim):
            combination = tuple(
                sorted((index, j))
            )  # Using a tuple to make combinations hashable
            if combination not in seen_combinations:
                seen_combinations.add(combination)
                poly_product(z, library, order - 1, j, latent_dim, seen_combinations)
                library.append(torch.prod(z[:, [index, j]], dim=1))


def sindy_library_simulate(z, poly_order, include_sine=False):
    # initialize library
    # append state variables
    m, n = z.shape
    latent_dim = n
    l = library_size(n, poly_order)

    library = [1.0]

    sample_list = range(latent_dim)
    list_combinations = list()

    for n in range(1, poly_order + 1):
        list_combinations = list(combinations_with_replacement(sample_list, n))
        for combination in list_combinations:
            library.append(np.prod(z[:, combination]))

    # add sine terms if included
    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:, i]))

    return np.stack(library, axis=0)


def sindy_library_tf(z, latent_dim, poly_order, include_sine=False):

    library = [torch.ones(z.shape[0]).to("cuda")]

    for i in range(latent_dim):
        library.append(z[:, i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                library.append(torch.multiply(z[:, i], z[:, j]))

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    library.append(z[:, i] * z[:, j] * z[:, k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        for q in range(p, latent_dim):
                            library.append(
                                z[:, i] * z[:, j] * z[:, k] * z[:, p] * z[:, q]
                            )

    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:, i]))

    return torch.stack(library, dim=1).to("cuda")


def sindy_simulate(x0, t, Xi, poly_order, include_sine):
    m = t.size
    n = x0.size
    f = lambda x, t: np.dot(
        sindy_library_simulate(np.array(x).reshape((1, n)), poly_order, include_sine),
        Xi,
    ).reshape((n,))

    x = odeint(f, x0, t)
    return x
