import numpy as np


# ============================================================
# Basic helpers
# ============================================================

def _as_float64(a):
    return np.asarray(a, dtype=np.float64)


def _check_main_inputs(x, u, d):
    x = _as_float64(x)
    u = _as_float64(u)
    d = _as_float64(d)

    if x.ndim != 1:
        raise ValueError("x must be 1-D with shape (Nt,)")

    if u.ndim != 3 or d.ndim != 3:
        raise ValueError("u and d must have shape (2, Nt, Nr)")

    if u.shape != d.shape:
        raise ValueError("u and d must have the same shape")

    if u.shape[0] != 2:
        raise ValueError("u and d must have first dimension size 2")

    if u.shape[1] != x.shape[0]:
        raise ValueError("u.shape[1] must match x.shape[0]")

    if x.shape[0] < 3:
        raise ValueError("x must contain at least three samples")

    dx = np.diff(x)
    if not np.all(dx > 0.0):
        raise ValueError("x must be strictly increasing")

    if not np.allclose(dx, dx[0], rtol=1e-10, atol=1e-14):
        raise ValueError("this implementation assumes a uniform time grid")

    return x, u, d


def trapezoid_keepdims(y, x, axis=0):
    return np.expand_dims(np.trapezoid(y, x=x, axis=axis), axis=axis)


def softplus_and_derivative(u, beta=2.0):
    beta = float(beta)
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    u = _as_float64(u)

    Ru = np.logaddexp(0.0, beta * u) / beta
    sigma = np.exp(-np.logaddexp(0.0, -beta * u))

    return Ru, sigma


# ============================================================
# Polar decomposition
# ============================================================

def radius_decomp(u):
    u = _as_float64(u)

    ux = u[0]
    uz = u[1]

    u_r = np.sqrt(ux * ux + uz * uz)

    if np.any(u_r <= 0.0):
        raise ValueError(
            "zero radius encountered in radius_decomp. "
            "For the FWI implementation, use the softplus wrapper."
        )

    cos_theta = ux / u_r
    sin_theta = uz / u_r

    u_theta = np.empty((2, 2, *u_r.shape), dtype=np.float64)

    u_theta[0, 0] = cos_theta
    u_theta[0, 1] = -sin_theta
    u_theta[1, 0] = sin_theta
    u_theta[1, 1] = cos_theta

    return u_r, u_theta


# ============================================================
# 1D OT helpers
# ============================================================

def cdf_batch(x, y):
    x = _as_float64(x)
    y = _as_float64(y)

    dx = x[1] - x[0]

    out = np.zeros_like(y, dtype=np.float64)
    out[1:, :] = np.cumsum(
        0.5 * (y[1:, :] + y[:-1, :]) * dx,
        axis=0,
    )

    return out


def opt_map_batch(x, F, G):
    x = _as_float64(x)
    F = _as_float64(F)
    G = _as_float64(G)

    Nt, Nr = F.shape
    out = np.empty((Nt, Nr), dtype=np.float64)

    for r in range(Nr):
        out[:, r] = np.interp(F[:, r], G[:, r], x)

    return out


def _uniform_interp_indices(x, values):
    x = _as_float64(x)
    values = _as_float64(values)

    dx = x[1] - x[0]
    x0 = x[0]
    Nt = x.shape[0]

    idx = np.floor((values - x0) / dx).astype(np.int64)
    idx = np.clip(idx, 0, Nt - 2)

    x_left = x0 + idx * dx
    w = (values - x_left) / dx

    return idx, w


def _interp_scalar_by_receiver(arr, idx, w):
    arr = _as_float64(arr)

    Nr = arr.shape[1]
    cols = np.arange(Nr)[None, :]

    return (
        (1.0 - w) * arr[idx, cols]
        + w * arr[idx + 1, cols]
    )


def _interp_time_matrix_by_receiver(arr, idx, w):
    arr = _as_float64(arr)

    idx4 = idx[None, None, :, :]
    w4 = w[None, None, :, :]

    low = np.take_along_axis(arr, idx4, axis=2)
    high = np.take_along_axis(arr, idx4 + 1, axis=2)

    return (1.0 - w4) * low + w4 * high


def _reverse_trapezoid_integral_uniform(x, integrand):
    x = _as_float64(x)
    integrand = _as_float64(integrand)

    dx = x[1] - x[0]
    Nt, Nr = integrand.shape

    segment = 0.5 * (integrand[:-1, :] + integrand[1:, :]) * dx

    out = np.zeros((Nt, Nr), dtype=np.float64)
    out[:-1, :] = np.cumsum(segment[::-1, :], axis=0)[::-1, :]

    return out


def int_helper_batch(x, u, v):
    x = _as_float64(x)
    u = _as_float64(u)
    v = _as_float64(v)

    dx = x[1] - x[0]
    Nt, Nr = u.shape

    integrand_segment = 0.5 * (
        (x[:-1, None] - v[:-1, :])
        + (x[1:, None] - v[1:, :])
    ) * dx

    prefix = np.zeros((Nt, Nr), dtype=np.float64)
    prefix[:-1, :] = np.cumsum(
        integrand_segment[::-1, :],
        axis=0,
    )[::-1, :]

    idx, w = _uniform_interp_indices(x, u)
    cols = np.arange(Nr)[None, :]

    return (
        (1.0 - w) * prefix[idx, cols]
        + w * prefix[idx + 1, cols]
    )


def W2_1d_batch_with_map(x, f, g):
    x = _as_float64(x)
    f = _as_float64(f)
    g = _as_float64(g)

    F = cdf_batch(x, f)
    G = cdf_batch(x, g)

    T_fg = opt_map_batch(x, F, G)
    T_gf = opt_map_batch(x, G, F)

    diff = x[:, None] - T_fg

    ot_vec = np.trapezoid(
        diff * diff * f,
        x=x,
        axis=0,
    )

    integral = int_helper_batch(
        x,
        T_fg,
        T_gf,
    )

    grad = diff * diff + 2.0 * integral

    return ot_vec, grad, T_fg


def W2_1d_batch(x, f, g):
    ot_vec, grad, _ = W2_1d_batch_with_map(x, f, g)
    return ot_vec, grad


# ============================================================
# ROT directional term
# ============================================================

def obj2_from_map(
    x,
    tu_r,
    d_r,
    u_theta,
    d_theta,
    T_map,
    density_floor=1e-14,
):
    x = _as_float64(x)
    tu_r = _as_float64(tu_r)
    d_r = _as_float64(d_r)
    u_theta = _as_float64(u_theta)
    d_theta = _as_float64(d_theta)
    T_map = _as_float64(T_map)

    idx, w = _uniform_interp_indices(x, T_map)

    # d_theta(T(t))
    d_theta_T = _interp_time_matrix_by_receiver(
        d_theta,
        idx,
        w,
    )

    dist = u_theta - d_theta_T

    # Frobenius norm squared
    dist_sq = np.einsum(
        "abtr,abtr->tr",
        dist,
        dist,
    )

    # J2
    J2 = np.sum(
        np.trapezoid(
            dist_sq * tu_r,
            x=x,
            axis=0,
        )
    )

    # Direct derivative with respect to u_theta
    grad2_theta = (
        2.0
        * dist
        * tu_r[None, None, :, :]
    )

    # d_theta'(T(t))
    dx = x[1] - x[0]

    d_theta_t = np.gradient(
        d_theta,
        dx,
        axis=2,
        edge_order=2,
    )

    d_theta_t_T = _interp_time_matrix_by_receiver(
        d_theta_t,
        idx,
        w,
    )

    inner_product = np.einsum(
        "abtr,abtr->tr",
        dist,
        d_theta_t_T,
    )

    # T'(t) = tu_r(t) / d_r(T(t))
    d_r_T = _interp_scalar_by_receiver(
        d_r,
        idx,
        w,
    )

    d_r_T = np.maximum(
        d_r_T,
        float(density_floor),
    )

    T_prime = tu_r / d_r_T

    # Since dist = u_theta - d_theta(T),
    #
    # 2<Target-source,target'> =
    # -2<dist,target'>
    transport_integrand = (
        -2.0
        * T_prime
        * inner_product
    )

    grad_helper = _reverse_trapezoid_integral_uniform(
        x,
        transport_integrand,
    )

    grad2_tr = dist_sq + grad_helper

    return J2, grad2_tr, grad2_theta


# ============================================================
# Main ROT objective and gradient
# ============================================================

def ot_dist_equalize_multitrace_with_grad(
    x,
    u,
    d,
    alpha=1.0,
):
    x, u, d = _check_main_inputs(x, u, d)
    alpha = float(alpha)

    ux = u[0]
    uz = u[1]

    # Radial/directional decomposition
    u_r, u_theta = radius_decomp(u)
    d_r, d_theta = radius_decomp(d)

    # Mass equalization, using the same trapezoidal rule
    # as the OT/CDF discretization.
    mass_u = trapezoid_keepdims(
        u_r,
        x,
        axis=0,
    )

    mass_d = trapezoid_keepdims(
        d_r,
        x,
        axis=0,
    )

    if np.any(mass_u <= 0.0):
        raise ValueError("zero radial mass in predicted trace")

    if np.any(mass_d <= 0.0):
        raise ValueError("zero radial mass in observed trace")

    tu_r = u_r * (mass_d / mass_u)

    # J1 and transport map
    ot_vec, grad1_tr, T_map = W2_1d_batch_with_map(
        x,
        tu_r,
        d_r,
    )

    J1 = np.sum(ot_vec)

    # J2
    J2, grad2_tr, grad2_theta = obj2_from_map(
        x,
        tu_r,
        d_r,
        u_theta,
        d_theta,
        T_map,
    )

    # dJ / d tu_r
    grad_tr = grad1_tr + alpha * grad2_tr

    # Chain rule through
    # tu_r = u_r * mass_d / mass_u.
    dot_term = trapezoid_keepdims(
        grad_tr * tu_r,
        x,
        axis=0,
    )

    grad_r = (
        mass_d * grad_tr
        - dot_term
    ) / mass_u

    # Chain rule through
    # u_r = sqrt(ux^2 + uz^2)
    grad_radial_u = (
        grad_r[None, :, :]
        * (u / u_r[None, :, :])
    )

    # Chain rule through u_theta.
    inv_r = 1.0 / u_r
    inv_r2 = inv_r * inv_r

    A00 = uz * inv_r
    A01 = ux * inv_r
    A10 = -ux * inv_r
    A11 = uz * inv_r

    contraction = (
        grad2_theta[0, 0] * A00
        + grad2_theta[0, 1] * A01
        + grad2_theta[1, 0] * A10
        + grad2_theta[1, 1] * A11
    )

    grad_theta_u = np.empty_like(u)

    grad_theta_u[0] = (
        uz * inv_r2 * contraction
    )

    grad_theta_u[1] = (
        -ux * inv_r2 * contraction
    )

    J = J1 + alpha * J2

    grad = (
        grad_radial_u
        + alpha * grad_theta_u
    )

    return J, grad


def ot_dist_equalize_multitrace_with_grad_softplus_encoding(
    x,
    u,
    d,
    alpha=1.0,
    beta=2.0,
):
    '''
    Main ROT function.
    '''
    x, u, d = _check_main_inputs(x, u, d)

    Ru, sigma_u = softplus_and_derivative(
        u,
        beta=beta,
    )

    Rd, _ = softplus_and_derivative(
        d,
        beta=beta,
    )

    J, grad_encoded = ot_dist_equalize_multitrace_with_grad(
        x,
        Ru,
        Rd,
        alpha=alpha,
    )

    grad = grad_encoded * sigma_u

    return J, grad


# ============================================================
# Backward-compatible helper names
# ============================================================

def obj1(x, tu_r, d_r):
    obj_raw, grad1_tr, _ = W2_1d_batch_with_map(
        _as_float64(x),
        _as_float64(tu_r),
        _as_float64(d_r),
    )
    return np.sum(obj_raw), grad1_tr


def obj2(x, tu_r, d_r, u_theta, d_theta):
    x = _as_float64(x)
    tu_r = _as_float64(tu_r)
    d_r = _as_float64(d_r)

    F = cdf_batch(x, tu_r)
    G = cdf_batch(x, d_r)
    T_map = opt_map_batch(x, F, G)

    return obj2_from_map(
        x,
        tu_r,
        d_r,
        u_theta,
        d_theta,
        T_map,
    )
