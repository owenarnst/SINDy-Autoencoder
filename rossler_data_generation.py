import numpy as np
from scipy.integrate import odeint
from scipy.special import legendre, chebyt
from SINDy_library import library_size


def get_rossler_data(n_ics, noise_strength=0):
    """
    Generate a set of Rössler training data for multiple random initial conditions.

    Arguments:
        n_ics - Integer specifying the number of initial conditions to use.
        noise_strength - Amount of noise to add to the data.

    Returns:
        data - Dictionary containing time series data and spatial embeddings.
    """
    t = np.arange(0, 5, 0.02)
    n_steps = t.size
    input_dim = 128

    ic_means = np.array([0, 0, 0])  # Centered near origin
    ic_widths = 2 * np.array([10, 10, 10])  # Widths can be tuned later

    # Create random initial conditions
    ics = ic_widths * (np.random.rand(n_ics, 3) - 0.5) + ic_means

    data = generate_rossler_data(
        ics,
        t,
        input_dim,
        linear=False,
        normalization=np.array([1 / 40, 1 / 40, 1 / 40]),  # Same as Lorenz, for now
    )

    # Reshape and add noise
    data["x"] = data["x"].reshape((-1, input_dim)) + noise_strength * np.random.randn(
        n_steps * n_ics, input_dim
    )
    data["dx"] = data["dx"].reshape((-1, input_dim)) + noise_strength * np.random.randn(
        n_steps * n_ics, input_dim
    )
    data["ddx"] = data["ddx"].reshape((-1, input_dim)) + noise_strength * np.random.randn(
        n_steps * n_ics, input_dim
    )

    return data


def simulate_rossler(z0, t, a=0.2, b=0.2, c=5.7):
    """
    Simulate the Rössler dynamics.

    Arguments:
        z0 - Initial condition in the form of a 3-value list or array.
        t - Array of time points at which to simulate.
        a, b, c - Rössler parameters

    Returns:
        z, dz, ddz - Arrays of the trajectory values and their 1st and 2nd derivatives.
    """
    # Define first derivative
    f = lambda z, t: [
        -z[1] - z[2],
        z[0] + a * z[1],
        b + z[2] * (z[0] - c),
    ]

    # Define second derivative (chain rule applied manually)
    df = lambda z, dz, t: [
        -dz[1] - dz[2],
        dz[0] + a * dz[1],
        dz[2] * (z[0] - c) + z[2] * dz[0],  # product rule applied
    ]

    z = odeint(f, z0, t)

    dt = t[1] - t[0]
    dz = np.zeros(z.shape)
    ddz = np.zeros(z.shape)
    for i in range(t.size):
        dz[i] = f(z[i], dt * i)
        ddz[i] = df(z[i], dz[i], dt * i)

    return z, dz, ddz


def generate_rossler_data(
    ics, t, n_points, linear=True, normalization=None
):
    """
    Generate high-dimensional Rössler dataset.

    Arguments:
        ics - Nx3 array of N initial conditions
        t - array of time points over which to simulate
        n_points - size of the high-dimensional dataset created
        linear - If True, use only linear spatial modes; if False, include cubic modes
        normalization - Optional 3-element array for scaling each Rössler variable

    Returns:
        data - Dictionary with time points (t), spatial basis (y_spatial), spatial modes (modes),
               latent states and derivatives (z, dz, ddz), high-dimensional embedding (x, dx, ddx),
               and a placeholder for sindy_coefficients.
    """

    n_ics = ics.shape[0]
    n_steps = t.size
    dt = t[1] - t[0]

    d = 3
    z = np.zeros((n_ics, n_steps, d))
    dz = np.zeros(z.shape)
    ddz = np.zeros(z.shape)

    for i in range(n_ics):
        z[i], dz[i], ddz[i] = simulate_rossler(ics[i], t)

    if normalization is not None:
        z *= normalization
        dz *= normalization
        ddz *= normalization

    n = n_points
    L = 1
    y_spatial = np.linspace(-L, L, n)

    modes = np.zeros((2 * d, n))
    for i in range(2 * d):
        modes[i] = legendre(i)(y_spatial)

    x = np.zeros((n_ics, n_steps, n))
    dx = np.zeros_like(x)
    ddx = np.zeros_like(x)

    for i in range(n_ics):
        for j in range(n_steps):
            z0, z1, z2 = z[i, j]
            dz0, dz1, dz2 = dz[i, j]
            ddz0, ddz1, ddz2 = ddz[i, j]

            x_lin = modes[0] * z0 + modes[1] * z1 + modes[2] * z2
            dx_lin = modes[0] * dz0 + modes[1] * dz1 + modes[2] * dz2
            ddx_lin = modes[0] * ddz0 + modes[1] * ddz1 + modes[2] * ddz2

            if linear:
                x[i, j] = x_lin
                dx[i, j] = dx_lin
                ddx[i, j] = ddx_lin
            else:
                x_nonlin = modes[3] * z0**3 + modes[4] * z1**3 + modes[5] * z2**3
                dx_nonlin = (
                    modes[3] * 3 * z0**2 * dz0 +
                    modes[4] * 3 * z1**2 * dz1 +
                    modes[5] * 3 * z2**2 * dz2
                )
                ddx_nonlin = (
                    modes[3] * (6 * z0 * dz0**2 + 3 * z0**2 * ddz0) +
                    modes[4] * (6 * z1 * dz1**2 + 3 * z1**2 * ddz1) +
                    modes[5] * (6 * z2 * dz2**2 + 3 * z2**2 * ddz2)
                )

                x[i, j] = x_lin + x_nonlin
                dx[i, j] = dx_lin + dx_nonlin
                ddx[i, j] = ddx_lin + ddx_nonlin

    data = {
        "t": t,
        "y_spatial": y_spatial,
        "modes": modes,
        "x": x,
        "dx": dx,
        "ddx": ddx,
        "z": z,
        "dz": dz,
        "ddz": ddz,
        "sindy_coefficients": None  # No analytic coefficients for Rössler
    }

    return data


if __name__ == "__main__":

    data = get_rossler_data(10, 0)

    print(data["x"].shape)
    print(data["dx"].shape)
    print(data["ddx"].shape)
    print(data["z"].shape)
    print(data["dz"].shape)
    print(data["ddz"].shape)
