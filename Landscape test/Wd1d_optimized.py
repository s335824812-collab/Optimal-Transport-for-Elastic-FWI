import numpy as np


def _check_inputs(x, u, d):
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("x must be 1-D with shape (Nt,)")
    if u.ndim != 2 or d.ndim != 2:
        raise ValueError("u and d must be 2-D with shape (Nt, Nr)")
    if u.shape != d.shape:
        raise ValueError("u and d must have the same shape")
    if x.shape[0] != u.shape[0]:
        raise ValueError("x.shape[0] must match u.shape[0]")

    return x, u, d


def _activation_and_derivative(u, kind="softplus", beta=10.0):
    if kind == "softplus":
        v = np.logaddexp(0.0, beta * u) / abs(beta)
        sigma = np.exp(-np.logaddexp(0.0, -beta * u))
    elif kind == "relu":
        v = np.maximum(u, 0.0)
        sigma = (u > 0.0).astype(np.float64)
    elif kind == "square":
        v = u * u
        sigma = 2.0 * u
    elif kind == "none":
        if np.min(u) < 0.0:
            raise ValueError("u must be non-negative for type='none'")
        v = u
        sigma = 1.0
    else:
        raise NotImplementedError(f"Unknown encoding type: {kind}")

    return v, sigma


def encoding(u, type="none", beta=10.0, dt=0.001):
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError("u must be 2-D with shape (Nt, Nr)")

    v, _ = _activation_and_derivative(u, kind=type, beta=beta)
    mass = np.sum(v, axis=0, keepdims=True) * dt

    if np.any(mass == 0.0):
        raise ValueError("zero mass encountered in encoding")

    return v / mass


def encoding_equalize(u, ref, type="none", beta=10.0, dt=0.001):
    u = np.asarray(u, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    if u.ndim != 2 or ref.ndim != 2:
        raise ValueError("u and ref must be 2-D with shape (Nt, Nr)")
    if u.shape != ref.shape:
        raise ValueError("u and ref must have the same shape")

    v_u, _ = _activation_and_derivative(u, kind=type, beta=beta)
    v_r, _ = _activation_and_derivative(ref, kind=type, beta=beta)

    mass_u = np.sum(v_u, axis=0, keepdims=True) * dt
    mass_r = np.sum(v_r, axis=0, keepdims=True) * dt

    if np.any(mass_u == 0.0):
        raise ValueError("zero mass encountered in u during equalization")

    return v_u * (mass_r / mass_u), v_r


def cdf_batch(x, y):
    dt = x[1] - x[0]

    out = np.empty_like(y, dtype=np.float64)
    out[0, :] = 0.0
    out[1:, :] = np.cumsum(0.5 * (y[1:, :] + y[:-1, :]) * dt, axis=0)

    return out


def opt_map(x, F, G):
    m = 0
    N = len(x)
    s = np.zeros_like(F, dtype=np.float64)

    for k in range(N):
        Ft = F[k]
        while G[m] < Ft:
            if m < N - 1:
                m += 1
            else:
                break

        if m == 0:
            s[k] = x[m]
        elif m == N - 1 and G[m] < Ft:
            s[k] = x[m]
        else:
            denom = G[m] - G[m - 1]
            if denom == 0.0:
                s[k] = x[m]
            else:
                alpha = (G[m] - Ft) / denom
                s[k] = alpha * x[m - 1] + (1.0 - alpha) * x[m]

    return s


def opt_map_batch(x, F, G):
    Nt, Nr = F.shape
    out = np.empty((Nt, Nr), dtype=np.float64)

    for r in range(Nr):
        out[:, r] = np.interp(F[:, r], G[:, r], x)

    return out


def int_helper(x, u, v):
    N = len(x)
    dx = x[1] - x[0]
    m = N - 2
    s = 0.0
    integral = np.zeros(N, dtype=np.float64)

    for k in range(N - 1, -1, -1):
        fc = u[k]

        while x[m] > fc and m > 0:
            s += 0.5 * ((x[m] - v[m]) + (x[m + 1] - v[m + 1])) * dx
            m -= 1

        if m == 0:
            integral[k] = s
        else:
            a = (fc - x[m]) / dx
            integral[k] = (
                s
                + 0.5
                * ((1.0 - a) * (x[m] - v[m]) + (1.0 + a) * (x[m + 1] - v[m + 1]))
                * (x[m + 1] - fc)
            )

    return integral


def int_helper_batch(x, u, v):
    Nt, Nr = u.shape
    dx = x[1] - x[0]
    x0 = x[0]

    integrand = 0.5 * (
        (x[:-1, None] - v[:-1, :]) +
        (x[1:, None] - v[1:, :])
    ) * dx

    prefix = np.empty((Nt, Nr), dtype=np.float64)
    prefix[-1, :] = 0.0
    prefix[:-1, :] = np.cumsum(integrand[::-1, :], axis=0)[::-1, :]

    # Uniform-grid replacement of per-column searchsorted(x, u[:, r]).
    idx = np.floor((u - x0) / dx).astype(np.int64)
    idx = np.clip(idx, 0, Nt - 2)

    x_idx = x0 + idx * dx
    t = (u - x_idx) / dx

    cols = np.arange(Nr)[None, :]

    return (1.0 - t) * prefix[idx, cols] + t * prefix[idx + 1, cols]


def W2_1d(x, f, g):
    F = cdf_batch(x, f[:, None])[:, 0]
    G = cdf_batch(x, g[:, None])[:, 0]
    s = opt_map(x, F, G)
    ot = np.trapezoid((x - s) ** 2 * f, x)
    return ot, s


def W2_grad(x, f, g):
    F = cdf_batch(x, f[:, None])[:, 0]
    G = cdf_batch(x, g[:, None])[:, 0]

    u = opt_map(x, F, G)
    v = opt_map(x, G, F)

    integral = int_helper(x, u, v)
    return (u - x) ** 2 + 2.0 * integral


def W2_1d_batch(x, f, g):
    x = np.asarray(x, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)

    F = cdf_batch(x, f)
    G = cdf_batch(x, g)

    T_fg = opt_map_batch(x, F, G)
    T_gf = opt_map_batch(x, G, F)

    diff = x[:, None] - T_fg
    ot = np.trapezoid(diff * diff * f, x, axis=0)

    integral = int_helper_batch(x, T_fg, T_gf)
    grad = diff * diff + 2.0 * integral

    return ot, grad


def get_ot_derivative(x, u, d, type="softplus", beta=10.0):
    x, u, d = _check_inputs(x, u, d)
    dt = x[1] - x[0]

    v_u, sigma = _activation_and_derivative(u, kind=type, beta=beta)
    v_d, _ = _activation_and_derivative(d, kind=type, beta=beta)

    mass_u = np.sum(v_u, axis=0, keepdims=True) * dt
    mass_d = np.sum(v_d, axis=0, keepdims=True) * dt

    if np.any(mass_u == 0.0):
        raise ValueError("zero mass encountered in u")
    if np.any(mass_d == 0.0):
        raise ValueError("zero mass encountered in d")

    Ru = v_u / mass_u
    Rd = v_d / mass_d

    ot_vec, pot_grad = W2_1d_batch(x, Ru, Rd)

    dot_term = np.sum(pot_grad * Ru, axis=0, keepdims=True) * dt

    if type == "none":
        grad = (pot_grad - dot_term) / mass_u
    else:
        grad = sigma * (pot_grad - dot_term) / mass_u

    return np.sum(ot_vec), grad


def get_ot_derivative_equalize(x, u, d, type="softplus", beta=10.0):
    """
    Main COT function.

    Ru = A(u) * mass(A(d)) / mass(A(u))
    Rd = A(d)

    Returns
    -------
    J : float
        sum_r W_2^2(Ru[:, r], Rd[:, r])
    grad : ndarray, shape (Nt, Nr)
        Frechet derivative with respect to u.
    """
    x, u, d = _check_inputs(x, u, d)
    dt = x[1] - x[0]

    v_u, sigma = _activation_and_derivative(u, kind=type, beta=beta)
    v_d, _ = _activation_and_derivative(d, kind=type, beta=beta)

    mass_u = np.sum(v_u, axis=0, keepdims=True) * dt
    mass_d = np.sum(v_d, axis=0, keepdims=True) * dt

    if np.any(mass_u == 0.0):
        raise ValueError("zero mass encountered in u")
    if np.any(mass_d == 0.0):
        raise ValueError("zero mass encountered in d")

    Ru = v_u * (mass_d / mass_u)
    Rd = v_d

    ot_vec, pot_grad = W2_1d_batch(x, Ru, Rd)

    # Chain rule through Ru = A(u) * mass_d / mass_u.
    dot_term = np.sum(pot_grad * Ru, axis=0, keepdims=True) * dt

    if type == "none":
        grad = (mass_d * pot_grad - dot_term) / mass_u
    else:
        grad = sigma * (mass_d * pot_grad - dot_term) / mass_u

    return np.sum(ot_vec), grad
