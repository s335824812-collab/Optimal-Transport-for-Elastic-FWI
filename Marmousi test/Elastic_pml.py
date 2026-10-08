"""
Devito handwritten elastic CPML solver, refactored from the notebook
`devito_deepwave_with_abc_plots.ipynb`.

Design
------
Model
    Static spatial model, physical/extended grid, elastic parameters, and PML
    profile parameters.

Geometry
    Source/receiver coordinates, time axis, Ricker wavelet, and Devito
    source/receiver objects.  Default receivers record displacement ux/uz.

Solver
    Devito wave-equation state, displacement integration, optional sparse
    displacement saving, CPML auxiliary memory variables, staggered finite-
    difference equations, operator construction, reset, and forward run.

The numerical update intentionally follows the original notebook closely:
- extended Devito grid = physical domain + PML;
- origin shifted by -npml * spacing;
- first-order velocity-stress elastic system;
- manual 4th-order staggered derivatives;
- two-side CPML memories with ps=0/1 and local px/pz coordinates;
- separate node/half PML coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from devito import (
    NODE,
    ConditionalDimension,
    Dimension,
    Eq,
    Function,
    Grid,
    Inc,
    Operator,
    TensorTimeFunction,
    VectorTimeFunction,
    initialize_function,
)
from devito import configuration
from examples.seismic import Receiver, RickerSource, TimeAxis, PointSource

ArrayLike = Union[float, int, np.ndarray]
CoordArray = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]
ReceiverSpec = Union[CoordArray, Mapping[str, CoordArray]]


# -----------------------------------------------------------------------------
# Small numerical helpers
# -----------------------------------------------------------------------------


def _as_dtype(dtype):
    """Return a NumPy scalar type accepted by Devito, e.g. np.float32.

    Devito internally calls ``dtype(value)`` in several places, so passing
    ``np.dtype("float32")`` breaks on recent NumPy versions.
    """
    if isinstance(dtype, np.dtype):
        return dtype.type
    if isinstance(dtype, str):
        return np.dtype(dtype).type
    if isinstance(dtype, type):
        return dtype
    return np.dtype(dtype).type


def _as_physical_array(value: ArrayLike, shape: Tuple[int, int], dtype) -> np.ndarray:
    """Return a 2-D physical-domain array with shape=(nx, nz)."""
    dtype = _as_dtype(dtype)
    arr = np.asarray(value, dtype=dtype)

    if arr.ndim == 0:
        return np.full(shape, arr.item(), dtype=dtype)

    if arr.shape != shape:
        raise ValueError(f"Expected array shape {shape}, got {arr.shape}.")

    return np.ascontiguousarray(arr, dtype=dtype)


def _as_2d_coordinates(coords: CoordArray, *, name: str) -> np.ndarray:
    """Normalize coordinate input to shape=(npoint, 2)."""
    arr = np.asarray(coords, dtype=np.float64)

    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError(f"{name} must be a length-2 coordinate or an (n, 2) array.")
        arr = arr.reshape(1, 2)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2), got {arr.shape}.")

    return np.ascontiguousarray(arr, dtype=np.float64)


def pml_fraction_1d(axis: np.ndarray, inner0: float, inner1: float, lpml: float) -> np.ndarray:
    """Normalized distance into the PML: 0 in the physical domain, 1 at outer edge."""
    if lpml <= 0.0:
        return np.zeros_like(axis, dtype=np.float64)
    return np.clip(np.maximum((inner0 - axis) / lpml, (axis - inner1) / lpml), 0.0, 1.0)


def make_pml_ab_1d(
    sigma: np.ndarray,
    alpha: np.ndarray,
    frac: np.ndarray,
    dt: float,
    tol: float,
    dtype,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct 1-D CPML memory coefficients for kappa=1.

    The memory update is

        memory_new = a * memory_old + b * derivative

    Coefficients outside the PML are set to zero.  The memory variables are only
    updated in PML loops, but zeroing the physical-domain coefficients makes the
    arrays safe to inspect/reuse.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    frac = np.asarray(frac, dtype=np.float64)

    a = np.zeros_like(sigma, dtype=np.float64)
    b = np.zeros_like(sigma, dtype=np.float64)

    inside = frac > 0.0
    denom = sigma[inside] + alpha[inside]
    a[inside] = np.exp(-denom * float(dt))
    b[inside] = sigma[inside] / (denom + float(tol)) * (a[inside] - 1.0)

    return a.astype(dtype), b.astype(dtype)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


@dataclass
class PMLConfig:
    """Static CPML profile parameters."""

    npml: int = 40
    r_val: float = 1.0e-5
    n_power: int = 3
    tol: float = 1.0e-8
    alpha0: Optional[float] = None
    # If not None, use this fixed velocity to construct CPML damping
    # instead of the current model's max(vp).  This is useful in FWI so
    # that the boundary operator does not change with the inverted model.
    vpmax: Optional[float] = 5.0

    def __post_init__(self) -> None:
        if self.npml < 0:
            raise ValueError("npml must be non-negative.")
        if self.r_val <= 0.0:
            raise ValueError("r_val must be positive.")
        if self.n_power <= 0:
            raise ValueError("n_power must be positive.")
        if self.vpmax is not None and float(self.vpmax) <= 0.0:
            raise ValueError("vpmax must be positive when provided.")


@dataclass
class ElasticModel:
    """
    Spatial elastic model and CPML grid information.

    Parameters are defined on the physical domain with shape=(nx, nz).  The
    Devito grid created here is the extended grid including PML layers.
    """

    vp: ArrayLike
    vs: ArrayLike
    rho: ArrayLike
    shape: Tuple[int, int]
    spacing: Tuple[float, float]
    origin: Tuple[float, float] = (0.0, 0.0)
    pml: PMLConfig = field(default_factory=PMLConfig)
    dtype: object = np.float32
    grid: Grid = field(init=False)

    # physical-domain arrays
    vp_array: np.ndarray = field(init=False)
    vs_array: np.ndarray = field(init=False)
    rho_array: np.ndarray = field(init=False)
    lam_array: np.ndarray = field(init=False)
    mu_array: np.ndarray = field(init=False)
    buoyancy_array: np.ndarray = field(init=False)

    # derived geometry
    extended_shape: Tuple[int, int] = field(init=False)
    extended_origin: Tuple[float, float] = field(init=False)
    physical_extent: Tuple[float, float] = field(init=False)
    extended_extent: Tuple[float, float] = field(init=False)
    x_axis: np.ndarray = field(init=False)
    z_axis: np.ndarray = field(init=False)
    x_axis_half: np.ndarray = field(init=False)
    z_axis_half: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.dtype = _as_dtype(self.dtype)
        self.shape = tuple(int(v) for v in self.shape)
        if len(self.shape) != 2:
            raise ValueError("shape must be (nx, nz).")
        if self.shape[0] <= 0 or self.shape[1] <= 0:
            raise ValueError("shape values must be positive.")

        self.spacing = tuple(float(v) for v in self.spacing)
        if len(self.spacing) != 2 or self.spacing[0] <= 0.0 or self.spacing[1] <= 0.0:
            raise ValueError("spacing must be positive (dx, dz).")

        self.origin = tuple(float(v) for v in self.origin)
        if len(self.origin) != 2:
            raise ValueError("origin must be (x0, z0).")

        self.vp_array = _as_physical_array(self.vp, self.shape, self.dtype)
        self.vs_array = _as_physical_array(self.vs, self.shape, self.dtype)
        self.rho_array = _as_physical_array(self.rho, self.shape, self.dtype)

        self.mu_array = self.rho_array * self.vs_array**2
        self.lam_array = self.rho_array * (self.vp_array**2 - 2.0 * self.vs_array**2)
        self.buoyancy_array = 1.0 / self.rho_array

        nx, nz = self.shape
        dx, dz = self.spacing
        npml = self.pml.npml

        self.extended_shape = (nx + 2 * npml, nz + 2 * npml)
        self.extended_origin = (self.origin[0] - npml * dx, self.origin[1] - npml * dz)

        # The original notebook uses extent=(n-1)*d + 2*npml*d with shape=n+2*npml.
        self.physical_extent = ((nx - 1) * dx, (nz - 1) * dz)
        self.extended_extent = (
            self.physical_extent[0] + 2 * npml * dx,
            self.physical_extent[1] + 2 * npml * dz,
        )

        x0, z0 = self.extended_origin
        self.x_axis = np.linspace(x0, x0 + self.extended_extent[0], self.extended_shape[0])
        self.z_axis = np.linspace(z0, z0 + self.extended_extent[1], self.extended_shape[1])
        self.x_axis_half = self.x_axis + dx / 2.0
        self.z_axis_half = self.z_axis + dz / 2.0

        self.grid = Grid(
            origin=self.extended_origin,
            extent=self.extended_extent,
            shape=self.extended_shape,
            dtype=self.dtype,
        )

    @property
    def vp_max(self) -> float:
        return float(np.max(self.vp_array))

    @property
    def vpmax_for_pml(self) -> float:
        """Velocity used only to build CPML damping coefficients.

        If ``pml.vpmax`` is provided, use it as a fixed PML velocity; otherwise
        fall back to the current model maximum velocity for backward
        compatibility.  ``check_time_step`` still uses ``vp_max`` from the
        actual model, because CFL should be checked against the current model.
        """
        return self.vp_max if self.pml.vpmax is None else float(self.pml.vpmax)

    @property
    def physical_bounds(self) -> Tuple[float, float, float, float]:
        """Return (x_inner0, x_inner1, z_inner0, z_inner1)."""
        x0, z0 = self.origin
        dx, dz = self.spacing
        nx, nz = self.shape
        return x0, x0 + (nx - 1) * dx, z0, z0 + (nz - 1) * dz

    def check_time_step(self, dt: float, cfl: float = 0.4, safety: float = 0.99) -> None:
        """Raise if dt exceeds the notebook-style CFL estimate."""
        dt_max = min(self.spacing) * float(cfl) / self.vp_max
        if float(dt) > float(safety) * dt_max:
            raise ValueError(f"dt={dt} is larger than {safety} * dt_max={safety * dt_max}.")

    def cpml_coefficient_arrays(self, dt: float, f0: float, dtype=None) -> Dict[str, np.ndarray]:
        """
        Build node/half CPML coefficient arrays.

        These arrays are static for a given (model, dt, f0), but are returned as
        NumPy arrays so Solver can own the Devito Function objects.
        """
        dtype = _as_dtype(dtype or self.dtype)
        npml = self.pml.npml
        dx, dz = self.spacing
        x_inner0, x_inner1, z_inner0, z_inner1 = self.physical_bounds
        lx_pml = npml * dx
        lz_pml = npml * dz

        x_frac = pml_fraction_1d(self.x_axis, x_inner0, x_inner1, lx_pml)
        z_frac = pml_fraction_1d(self.z_axis, z_inner0, z_inner1, lz_pml)
        x_frac_half = pml_fraction_1d(self.x_axis_half, x_inner0, x_inner1, lx_pml)
        z_frac_half = pml_fraction_1d(self.z_axis_half, z_inner0, z_inner1, lz_pml)

        if npml == 0:
            zero_x = np.zeros(self.extended_shape[0], dtype=dtype)
            zero_z = np.zeros(self.extended_shape[1], dtype=dtype)
            return {
                "ax_node": zero_x.copy(), "bx_node": zero_x.copy(),
                "ax_half": zero_x.copy(), "bx_half": zero_x.copy(),
                "az_node": zero_z.copy(), "bz_node": zero_z.copy(),
                "az_half": zero_z.copy(), "bz_half": zero_z.copy(),
            }

        vpmax_pml = self.vpmax_for_pml
        sigma0_x = -(1.0 + self.pml.n_power) * vpmax_pml * np.log(self.pml.r_val) / (2.0 * lx_pml)
        sigma0_z = -(1.0 + self.pml.n_power) * vpmax_pml * np.log(self.pml.r_val) / (2.0 * lz_pml)
        alpha0 = float(self.pml.alpha0) if self.pml.alpha0 is not None else np.pi * float(f0)

        sigma_x = sigma0_x * x_frac**self.pml.n_power
        sigma_z = sigma0_z * z_frac**self.pml.n_power
        sigma_x_half = sigma0_x * x_frac_half**self.pml.n_power
        sigma_z_half = sigma0_z * z_frac_half**self.pml.n_power

        alpha_x = alpha0 * (1.0 - x_frac)
        alpha_z = alpha0 * (1.0 - z_frac)
        alpha_x_half = alpha0 * (1.0 - x_frac_half)
        alpha_z_half = alpha0 * (1.0 - z_frac_half)

        ax_node, bx_node = make_pml_ab_1d(sigma_x, alpha_x, x_frac, dt, self.pml.tol, dtype)
        ax_half, bx_half = make_pml_ab_1d(sigma_x_half, alpha_x_half, x_frac_half, dt, self.pml.tol, dtype)
        az_node, bz_node = make_pml_ab_1d(sigma_z, alpha_z, z_frac, dt, self.pml.tol, dtype)
        az_half, bz_half = make_pml_ab_1d(sigma_z_half, alpha_z_half, z_frac_half, dt, self.pml.tol, dtype)

        return {
            "ax_node": ax_node, "bx_node": bx_node,
            "ax_half": ax_half, "bx_half": bx_half,
            "az_node": az_node, "bz_node": bz_node,
            "az_half": az_half, "bz_half": bz_half,
        }


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------


@dataclass
class AcquisitionGeometry:
    """Source/receiver geometry and time axis."""

    model: ElasticModel
    src_positions: CoordArray
    rec_positions: ReceiverSpec
    t0: float
    tn: float
    dt: float
    f0: float
    dtype: object = np.float32
    interpolation: Optional[str] = None
    receiver_components: Tuple[str, ...] = ("ux", "uz", "p")

    time_axis: TimeAxis = field(init=False)
    src_p_x: RickerSource = field(init=False)
    src_p_z: RickerSource = field(init=False)
    receivers: Dict[str, Receiver] = field(init=False, default_factory=dict)
    src_coordinates: np.ndarray = field(init=False)
    rec_coordinates: Dict[str, np.ndarray] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.dtype = _as_dtype(self.dtype)
        self.t0 = float(self.t0)
        self.tn = float(self.tn)
        self.dt = float(self.dt)
        self.f0 = float(self.f0)

        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.tn < self.t0:
            raise ValueError("tn must be >= t0.")

        self.nt = int((self.tn - self.t0) / self.dt) + 1
        self.time_axis = TimeAxis(start=self.t0, stop=self.tn, num=self.nt)

        self.src_coordinates = _as_2d_coordinates(self.src_positions, name="src_positions")
        self.rec_coordinates = self._normalize_receiver_coordinates(self.rec_positions)

        self._build_sources()
        self._build_receivers()

    def _source_kwargs(self, name: str, npoint: int) -> Dict[str, object]:
        kwargs: Dict[str, object] = dict(
            name=name,
            grid=self.model.grid,
            f0=self.f0,
            npoint=npoint,
            time_range=self.time_axis,
            dtype=self.dtype,
        )
        if self.interpolation is not None:
            kwargs["interpolation"] = self.interpolation
        return kwargs

    def _receiver_kwargs(self, name: str, npoint: int) -> Dict[str, object]:
        kwargs: Dict[str, object] = dict(
            name=name,
            grid=self.model.grid,
            npoint=npoint,
            time_range=self.time_axis,
            dtype=self.dtype,
        )
        if self.interpolation is not None:
            kwargs["interpolation"] = self.interpolation
        return kwargs

    def _normalize_receiver_coordinates(self, rec_positions: ReceiverSpec) -> Dict[str, np.ndarray]:
        if isinstance(rec_positions, Mapping):
            out = {}
            for comp in self.receiver_components:
                if comp not in rec_positions:
                    raise ValueError(f"rec_positions mapping is missing component {comp!r}.")
                out[comp] = _as_2d_coordinates(rec_positions[comp], name=f"rec_positions[{comp!r}]")
            return out

        coords = _as_2d_coordinates(rec_positions, name="rec_positions")
        return {comp: coords.copy() for comp in self.receiver_components}

    def _build_sources(self) -> None:
        nsrc = self.src_coordinates.shape[0]
        self.src_p_x = RickerSource(**self._source_kwargs("src_p_x", nsrc))
        self.src_p_z = RickerSource(**self._source_kwargs("src_p_z", nsrc))
        self.src_p_x.coordinates.data[:, :] = self.src_coordinates
        self.src_p_z.coordinates.data[:, :] = self.src_coordinates

    def _build_receivers(self) -> None:
        self.receivers = {}
        for comp in self.receiver_components:
            coords = self.rec_coordinates[comp]
            rec = Receiver(**self._receiver_kwargs(f"rec_{comp}", coords.shape[0]))
            rec.coordinates.data[:, :] = coords
            self.receivers[comp] = rec

    @property
    def rec_ux(self) -> Optional[Receiver]:
        return self.receivers.get("ux")

    @property
    def rec_uz(self) -> Optional[Receiver]:
        return self.receivers.get("uz")

    # Backward-compatible aliases.  These exist only if receiver_components
    # explicitly contains "vx"/"vz".
    @property
    def rec_vx(self) -> Optional[Receiver]:
        return self.receivers.get("vx")

    @property
    def rec_vz(self) -> Optional[Receiver]:
        return self.receivers.get("vz")

    @property
    def rec_p(self) -> Optional[Receiver]:
        return self.receivers.get("p")


# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------


@dataclass
class ElasticCPMLSolver:
    """
    Forward elastic velocity-stress solver with handwritten CPML.

    Model and Geometry are deliberately kept outside this class so that future
    adjoint/Born/gradient solvers can reuse them and replace only the PDE logic.
    """

    model: ElasticModel
    geometry: AcquisitionGeometry
    space_order: int = 4
    time_order: int = 1
    dtype: object = np.float32
    cfl: Optional[float] = 0.4
    save_u: bool = False
    u_save_interval: int = 10
    devito_opt: object = field(default_factory=lambda: ("advanced", {"cire-mingain": 50}))
    configure_devito: bool = True

    v: VectorTimeFunction = field(init=False)
    u: VectorTimeFunction = field(init=False)
    u_save: Optional[VectorTimeFunction] = field(init=False, default=None)
    coarse_time: Optional[ConditionalDimension] = field(init=False, default=None)
    n_u_save: Optional[int] = field(init=False, default=None)
    sigma: TensorTimeFunction = field(init=False)
    lam0: Function = field(init=False)
    mu0: Function = field(init=False)
    b0: Function = field(init=False)
    pml_memory_fields: Sequence[Function] = field(init=False)
    pml_coefficients: Dict[str, Function] = field(init=False, default_factory=dict)
    operator: Operator = field(init=False)

    def __post_init__(self) -> None:
        self.dtype = _as_dtype(self.dtype)
        if self.space_order != 4:
            raise NotImplementedError("This refactor currently preserves only the notebook's space_order=4 stencil.")
        if self.time_order != 1:
            raise NotImplementedError("This refactor currently preserves only the notebook's time_order=1 update.")
        if self.model.grid is not self.geometry.model.grid:
            raise ValueError("Geometry must be built from the same ElasticModel instance as Solver.")
        if self.cfl is not None:
            self.model.check_time_step(self.geometry.dt, cfl=float(self.cfl))
        if self.configure_devito:
            configuration["opt"] = "advanced"
            configuration["profiling"] = "advanced"

        self.grid = self.model.grid
        self.x, self.z = self.grid.dimensions
        self.t = self.grid.stepping_dim
        self.dt_symbol = self.grid.time_dim.spacing

        self.px = Dimension(name="px")
        self.pz = Dimension(name="pz")
        self.ps = Dimension(name="ps")
        self.LEFT = self.BOTTOM = 0
        self.RIGHT = self.TOP = 1
        self.x_left = self.px
        self.x_right = self.x.symbolic_max - self.px
        self.z_bottom = self.pz
        self.z_top = self.z.symbolic_max - self.pz

        self._build_fields()
        self._build_model_parameter_functions()
        self._build_pml_memory_fields()
        self._build_pml_coefficient_functions()
        self.operator = self._build_operator()
        self.reset()

    # ------------------------------------------------------------------
    # Field construction
    # ------------------------------------------------------------------

    def _build_fields(self) -> None:
        self.v = VectorTimeFunction(
            name="v",
            grid=self.grid,
            time_order=self.time_order,
            space_order=self.space_order,
            dtype=self.dtype,
        )
        self.u = VectorTimeFunction(
            name="u",
            grid=self.grid,
            time_order=self.time_order,
            space_order=self.space_order,
            dtype=self.dtype,
        )
        if self.save_u:
            if int(self.u_save_interval) <= 0:
                raise ValueError("u_save_interval must be positive when save_u=True.")
            self.u_save_interval = int(self.u_save_interval)
            # Number of actually written coarse-time snapshots.
            # With ConditionalDimension(factor=k), snapshots are written only
            # when the parent time satisfies the factor condition. Do not add
            # an extra slot, otherwise the last snapshot remains zero.
            self.n_u_save = 1 + (self.geometry.nt - 1) // self.u_save_interval
            self.coarse_time = ConditionalDimension(
                name="coarse_t",
                parent=self.grid.time_dim,
                factor=self.u_save_interval,
            )
            self.u_save = VectorTimeFunction(
                name="u_save",
                grid=self.grid,
                time_order=0,
                space_order=self.space_order,
                save=self.n_u_save,
                time_dim=self.coarse_time,
                dtype=self.dtype,
            )
        else:
            self.u_save = None
            self.coarse_time = None
            self.n_u_save = None

        self.sigma = TensorTimeFunction(
            name="sigma",
            grid=self.grid,
            time_order=self.time_order,
            space_order=self.space_order,
            dtype=self.dtype,
        )

    def _build_model_parameter_functions(self) -> None:
        npml = self.model.pml.npml
        self.lam0 = Function(name="lam0", grid=self.grid, space_order=self.space_order, dtype=self.dtype, parameter=True)
        initialize_function(self.lam0, self.model.lam_array, npml)

        self.mu0 = Function(
            name="mu0",
            grid=self.grid,
            space_order=self.space_order,
            dtype=self.dtype,
            parameter=True,
            # avg_mode="harmonic",
        )
        initialize_function(self.mu0, self.model.mu_array, npml)

        self.b0 = Function(name="b0", grid=self.grid, space_order=self.space_order, dtype=self.dtype, parameter=True)
        initialize_function(self.b0, self.model.buoyancy_array, npml)

    def _two_side_pml_memory(self, name, dimensions, shape, staggered=NODE) -> Function:
        return Function(
            name=name,
            dimensions=dimensions,
            shape=shape,
            space_order=self.space_order,
            dtype=self.dtype,
            staggered=staggered,
        )

    def _build_pml_memory_fields(self) -> None:
        npml = self.model.pml.npml
        nx_total, nz_total = self.model.extended_shape

        self.phi_sigmaxx_x = self._two_side_pml_memory(
            "phi_sigmaxx_x", dimensions=(self.ps, self.px, self.z), shape=(2, npml, nz_total), staggered=self.px
        )
        self.phi_sigmaxz_x = self._two_side_pml_memory(
            "phi_sigmaxz_x", dimensions=(self.ps, self.px, self.z), shape=(2, npml, nz_total), staggered=self.z
        )
        self.phi_sigmaxz_z = self._two_side_pml_memory(
            "phi_sigmaxz_z", dimensions=(self.ps, self.x, self.pz), shape=(2, nx_total, npml), staggered=self.x
        )
        self.phi_sigmazz_z = self._two_side_pml_memory(
            "phi_sigmazz_z", dimensions=(self.ps, self.x, self.pz), shape=(2, nx_total, npml), staggered=self.pz
        )

        self.phi_vx_x = self._two_side_pml_memory(
            "phi_vx_x", dimensions=(self.ps, self.px, self.z), shape=(2, npml, nz_total), staggered=NODE
        )
        self.phi_vz_x = self._two_side_pml_memory(
            "phi_vz_x", dimensions=(self.ps, self.px, self.z), shape=(2, npml, nz_total), staggered=(self.px, self.z)
        )
        self.phi_vz_z = self._two_side_pml_memory(
            "phi_vz_z", dimensions=(self.ps, self.x, self.pz), shape=(2, nx_total, npml), staggered=NODE
        )
        self.phi_vx_z = self._two_side_pml_memory(
            "phi_vx_z", dimensions=(self.ps, self.x, self.pz), shape=(2, nx_total, npml), staggered=(self.x, self.pz)
        )

        self.pml_memory_fields = [
            self.phi_sigmaxx_x, self.phi_sigmaxz_x,
            self.phi_sigmaxz_z, self.phi_sigmazz_z,
            self.phi_vx_x, self.phi_vz_x,
            self.phi_vz_z, self.phi_vx_z,
        ]

    def _make_1d_coeff_function(self, name: str, dimension, shape, staggered) -> Function:
        return Function(
            name=name,
            shape=shape,
            dimensions=(dimension,),
            space_order=0,
            parameter=True,
            dtype=self.dtype,
            staggered=staggered,
        )

    def _build_pml_coefficient_functions(self) -> None:
        coeff_arrays = self.model.cpml_coefficient_arrays(self.geometry.dt, self.geometry.f0, dtype=self.dtype)
        nx_total, nz_total = self.model.extended_shape

        self.pml_coefficients = {
            "ax_node": self._make_1d_coeff_function("ax_node", self.x, (nx_total,), NODE),
            "bx_node": self._make_1d_coeff_function("bx_node", self.x, (nx_total,), NODE),
            "ax_half": self._make_1d_coeff_function("ax_half", self.x, (nx_total,), self.x),
            "bx_half": self._make_1d_coeff_function("bx_half", self.x, (nx_total,), self.x),
            "az_node": self._make_1d_coeff_function("az_node", self.z, (nz_total,), NODE),
            "bz_node": self._make_1d_coeff_function("bz_node", self.z, (nz_total,), NODE),
            "az_half": self._make_1d_coeff_function("az_half", self.z, (nz_total,), self.z),
            "bz_half": self._make_1d_coeff_function("bz_half", self.z, (nz_total,), self.z),
        }

        for key, func in self.pml_coefficients.items():
            func.data[:] = coeff_arrays[key]

    # ------------------------------------------------------------------
    # Equation construction
    # ------------------------------------------------------------------

    def _at_x(self, expr, xidx):
        return expr.subs(self.x, xidx)

    def _at_z(self, expr, zidx):
        return expr.subs(self.z, zidx)

    def _build_staggered_derivatives(self) -> Dict[str, object]:
        x, z, t = self.x, self.z, self.t
        hx = x.spacing
        hz = z.spacing
        c1 = self.dtype(9.0 / 8.0)
        c2 = self.dtype(1.0 / 24.0)

        sxx = self.sigma[0, 0]
        sxz = self.sigma[0, 1]
        szz = self.sigma[1, 1]
        vx = self.v[0]
        vz = self.v[1]

        dsxx_dx = (c1 * (sxx[t, x + 1, z] - sxx[t, x, z]) - c2 * (sxx[t, x + 2, z] - sxx[t, x - 1, z])) / hx
        dsxz_dz = (c1 * (sxz[t, x, z] - sxz[t, x, z - 1]) - c2 * (sxz[t, x, z + 1] - sxz[t, x, z - 2])) / hz
        dsxz_dx = (c1 * (sxz[t, x, z] - sxz[t, x - 1, z]) - c2 * (sxz[t, x + 1, z] - sxz[t, x - 2, z])) / hx
        dszz_dz = (c1 * (szz[t, x, z + 1] - szz[t, x, z]) - c2 * (szz[t, x, z + 2] - szz[t, x, z - 1])) / hz

        # These use t+1 velocity values, exactly as in the notebook.
        dvx_dx = (c1 * (vx[t + 1, x, z] - vx[t + 1, x - 1, z]) - c2 * (vx[t + 1, x + 1, z] - vx[t + 1, x - 2, z])) / hx
        dvz_dz = (c1 * (vz[t + 1, x, z] - vz[t + 1, x, z - 1]) - c2 * (vz[t + 1, x, z + 1] - vz[t + 1, x, z - 2])) / hz
        dvx_dz = (c1 * (vx[t + 1, x, z + 1] - vx[t + 1, x, z]) - c2 * (vx[t + 1, x, z + 2] - vx[t + 1, x, z - 1])) / hz
        dvz_dx = (c1 * (vz[t + 1, x + 1, z] - vz[t + 1, x, z]) - c2 * (vz[t + 1, x + 2, z] - vz[t + 1, x - 1, z])) / hx

        return {
            "dsxx_dx": dsxx_dx, "dsxz_dz": dsxz_dz,
            "dsxz_dx": dsxz_dx, "dszz_dz": dszz_dz,
            "dvx_dx": dvx_dx, "dvz_dz": dvz_dz,
            "dvx_dz": dvx_dz, "dvz_dx": dvz_dx,
        }

    def _build_velocity_update(self, d: Dict[str, object]) -> Sequence[object]:
        dt = self.dt_symbol
        return [
            Eq(self.v[0].forward, self.v[0] + dt * self.b0 * (d["dsxx_dx"] + d["dsxz_dz"])),
            Eq(self.v[1].forward, self.v[1] + dt * self.b0 * (d["dsxz_dx"] + d["dszz_dz"])),
        ]

    def _build_displacement_update(self) -> Sequence[object]:
        """Integrate displacement from velocity: u.dt = v.

        The velocity update computes v.forward first, so we use v.forward here.
        This makes receiver interpolation of u.forward consistent with the old
        v.forward receiver timing.
        """
        dt = self.dt_symbol
        return [
            Eq(self.u[0].forward, self.u[0] + dt * self.v[0].forward),
            Eq(self.u[1].forward, self.u[1] + dt * self.v[1].forward),
        ]

    def _build_displacement_save(self) -> Sequence[object]:
        """Optionally save u.forward on a coarse conditional time grid."""
        if self.u_save is None:
            return []
        return [
            Eq(self.u_save[0], self.u[0].forward),
            Eq(self.u_save[1], self.u[1].forward),
        ]

    def _build_stress_to_velocity_cpml(self, d: Dict[str, object]) -> Sequence[object]:
        dt = self.dt_symbol
        ax_node = self.pml_coefficients["ax_node"]
        bx_node = self.pml_coefficients["bx_node"]
        ax_half = self.pml_coefficients["ax_half"]
        bx_half = self.pml_coefficients["bx_half"]
        az_node = self.pml_coefficients["az_node"]
        bz_node = self.pml_coefficients["bz_node"]
        az_half = self.pml_coefficients["az_half"]
        bz_half = self.pml_coefficients["bz_half"]

        x, z, px, pz = self.x, self.z, self.px, self.pz
        x_left, x_right = self.x_left, self.x_right
        z_bottom, z_top = self.z_bottom, self.z_top
        LEFT, RIGHT, BOTTOM, TOP = self.LEFT, self.RIGHT, self.BOTTOM, self.TOP
        eqs = []

        phi_sxx_x_L = self.phi_sigmaxx_x[LEFT, px, z]
        phi_sxz_x_L = self.phi_sigmaxz_x[LEFT, px, z]
        phi_sxx_x_R = self.phi_sigmaxx_x[RIGHT, px, z]
        phi_sxz_x_R = self.phi_sigmaxz_x[RIGHT, px, z]

        psi_sxx_x_L = ax_half[px] * phi_sxx_x_L + bx_half[px] * self._at_x(d["dsxx_dx"], x_left)
        psi_sxz_x_L = ax_node[px] * phi_sxz_x_L + bx_node[px] * self._at_x(d["dsxz_dx"], x_left)
        psi_sxx_x_R = ax_half[x_right] * phi_sxx_x_R + bx_half[x_right] * self._at_x(d["dsxx_dx"], x_right)
        psi_sxz_x_R = ax_node[x_right] * phi_sxz_x_R + bx_node[x_right] * self._at_x(d["dsxz_dx"], x_right)

        eqs += [
            Eq(phi_sxx_x_L, psi_sxx_x_L),
            Inc(self._at_x(self.v[0].forward, x_left), dt * self._at_x(self.b0, x_left) * phi_sxx_x_L),
            Eq(phi_sxz_x_L, psi_sxz_x_L),
            Inc(self._at_x(self.v[1].forward, x_left), dt * self._at_x(self.b0, x_left) * phi_sxz_x_L),
            Eq(phi_sxx_x_R, psi_sxx_x_R),
            Inc(self._at_x(self.v[0].forward, x_right), dt * self._at_x(self.b0, x_right) * phi_sxx_x_R),
            Eq(phi_sxz_x_R, psi_sxz_x_R),
            Inc(self._at_x(self.v[1].forward, x_right), dt * self._at_x(self.b0, x_right) * phi_sxz_x_R),
        ]

        phi_sxz_z_B = self.phi_sigmaxz_z[BOTTOM, x, pz]
        phi_szz_z_B = self.phi_sigmazz_z[BOTTOM, x, pz]
        phi_sxz_z_T = self.phi_sigmaxz_z[TOP, x, pz]
        phi_szz_z_T = self.phi_sigmazz_z[TOP, x, pz]

        psi_sxz_z_B = az_node[pz] * phi_sxz_z_B + bz_node[pz] * self._at_z(d["dsxz_dz"], z_bottom)
        psi_szz_z_B = az_half[pz] * phi_szz_z_B + bz_half[pz] * self._at_z(d["dszz_dz"], z_bottom)
        psi_sxz_z_T = az_node[z_top] * phi_sxz_z_T + bz_node[z_top] * self._at_z(d["dsxz_dz"], z_top)
        psi_szz_z_T = az_half[z_top] * phi_szz_z_T + bz_half[z_top] * self._at_z(d["dszz_dz"], z_top)

        eqs += [
            Eq(phi_sxz_z_B, psi_sxz_z_B),
            Inc(self._at_z(self.v[0].forward, z_bottom), dt * self._at_z(self.b0, z_bottom) * phi_sxz_z_B),
            Eq(phi_szz_z_B, psi_szz_z_B),
            Inc(self._at_z(self.v[1].forward, z_bottom), dt * self._at_z(self.b0, z_bottom) * phi_szz_z_B),
            Eq(phi_sxz_z_T, psi_sxz_z_T),
            Inc(self._at_z(self.v[0].forward, z_top), dt * self._at_z(self.b0, z_top) * phi_sxz_z_T),
            Eq(phi_szz_z_T, psi_szz_z_T),
            Inc(self._at_z(self.v[1].forward, z_top), dt * self._at_z(self.b0, z_top) * phi_szz_z_T),
        ]

        return eqs

    def _build_stress_update(self, d: Dict[str, object]) -> Sequence[object]:
        dt = self.dt_symbol
        return [
            Eq(
                self.sigma[0, 0].forward,
                self.sigma[0, 0] + dt * ((self.lam0 + 2 * self.mu0) * d["dvx_dx"] + self.lam0 * d["dvz_dz"]),
            ),
            Eq(
                self.sigma[1, 1].forward,
                self.sigma[1, 1] + dt * (self.lam0 * d["dvx_dx"] + (self.lam0 + 2 * self.mu0) * d["dvz_dz"]),
            ),
            Eq(
                self.sigma[0, 1].forward,
                self.sigma[0, 1] + dt * self.mu0 * (d["dvx_dz"] + d["dvz_dx"]),
            ),
        ]

    def _build_velocity_to_stress_cpml(self, d: Dict[str, object]) -> Sequence[object]:
        dt = self.dt_symbol
        ax_node = self.pml_coefficients["ax_node"]
        bx_node = self.pml_coefficients["bx_node"]
        ax_half = self.pml_coefficients["ax_half"]
        bx_half = self.pml_coefficients["bx_half"]
        az_node = self.pml_coefficients["az_node"]
        bz_node = self.pml_coefficients["bz_node"]
        az_half = self.pml_coefficients["az_half"]
        bz_half = self.pml_coefficients["bz_half"]

        x, z, px, pz = self.x, self.z, self.px, self.pz
        x_left, x_right = self.x_left, self.x_right
        z_bottom, z_top = self.z_bottom, self.z_top
        LEFT, RIGHT, BOTTOM, TOP = self.LEFT, self.RIGHT, self.BOTTOM, self.TOP
        eqs = []

        phi_vx_x_L = self.phi_vx_x[LEFT, px, z]
        phi_vz_x_L = self.phi_vz_x[LEFT, px, z]
        phi_vx_x_R = self.phi_vx_x[RIGHT, px, z]
        phi_vz_x_R = self.phi_vz_x[RIGHT, px, z]

        psi_vx_x_L = ax_node[px] * phi_vx_x_L + bx_node[px] * self._at_x(d["dvx_dx"], x_left)
        psi_vz_x_L = ax_half[px] * phi_vz_x_L + bx_half[px] * self._at_x(d["dvz_dx"], x_left)
        psi_vx_x_R = ax_node[x_right] * phi_vx_x_R + bx_node[x_right] * self._at_x(d["dvx_dx"], x_right)
        psi_vz_x_R = ax_half[x_right] * phi_vz_x_R + bx_half[x_right] * self._at_x(d["dvz_dx"], x_right)

        lam_L = self._at_x(self.lam0, x_left)
        mu_L = self._at_x(self.mu0, x_left)
        lam_R = self._at_x(self.lam0, x_right)
        mu_R = self._at_x(self.mu0, x_right)

        eqs += [
            Eq(phi_vx_x_L, psi_vx_x_L),
            Inc(self._at_x(self.sigma[0, 0].forward, x_left), dt * (lam_L + 2 * mu_L) * phi_vx_x_L),
            Inc(self._at_x(self.sigma[1, 1].forward, x_left), dt * lam_L * phi_vx_x_L),
            Eq(phi_vz_x_L, psi_vz_x_L),
            Inc(self._at_x(self.sigma[0, 1].forward, x_left), dt * mu_L * phi_vz_x_L),
            Eq(phi_vx_x_R, psi_vx_x_R),
            Inc(self._at_x(self.sigma[0, 0].forward, x_right), dt * (lam_R + 2 * mu_R) * phi_vx_x_R),
            Inc(self._at_x(self.sigma[1, 1].forward, x_right), dt * lam_R * phi_vx_x_R),
            Eq(phi_vz_x_R, psi_vz_x_R),
            Inc(self._at_x(self.sigma[0, 1].forward, x_right), dt * mu_R * phi_vz_x_R),
        ]

        phi_vz_z_B = self.phi_vz_z[BOTTOM, x, pz]
        phi_vx_z_B = self.phi_vx_z[BOTTOM, x, pz]
        phi_vz_z_T = self.phi_vz_z[TOP, x, pz]
        phi_vx_z_T = self.phi_vx_z[TOP, x, pz]

        psi_vz_z_B = az_node[pz] * phi_vz_z_B + bz_node[pz] * self._at_z(d["dvz_dz"], z_bottom)
        psi_vx_z_B = az_half[pz] * phi_vx_z_B + bz_half[pz] * self._at_z(d["dvx_dz"], z_bottom)
        psi_vz_z_T = az_node[z_top] * phi_vz_z_T + bz_node[z_top] * self._at_z(d["dvz_dz"], z_top)
        psi_vx_z_T = az_half[z_top] * phi_vx_z_T + bz_half[z_top] * self._at_z(d["dvx_dz"], z_top)

        lam_B = self._at_z(self.lam0, z_bottom)
        mu_B = self._at_z(self.mu0, z_bottom)
        lam_T = self._at_z(self.lam0, z_top)
        mu_T = self._at_z(self.mu0, z_top)

        eqs += [
            Eq(phi_vz_z_B, psi_vz_z_B),
            Inc(self._at_z(self.sigma[0, 0].forward, z_bottom), dt * lam_B * phi_vz_z_B),
            Inc(self._at_z(self.sigma[1, 1].forward, z_bottom), dt * (lam_B + 2 * mu_B) * phi_vz_z_B),
            Eq(phi_vx_z_B, psi_vx_z_B),
            Inc(self._at_z(self.sigma[0, 1].forward, z_bottom), dt * mu_B * phi_vx_z_B),
            Eq(phi_vz_z_T, psi_vz_z_T),
            Inc(self._at_z(self.sigma[0, 0].forward, z_top), dt * lam_T * phi_vz_z_T),
            Inc(self._at_z(self.sigma[1, 1].forward, z_top), dt * (lam_T + 2 * mu_T) * phi_vz_z_T),
            Eq(phi_vx_z_T, psi_vx_z_T),
            Inc(self._at_z(self.sigma[0, 1].forward, z_top), dt * mu_T * phi_vx_z_T),
        ]

        return eqs

    def _build_source_terms(self) -> Sequence[object]:
        dt = self.dt_symbol
        src_p_x_term = self.geometry.src_p_x.inject(field=self.sigma[0, 0].forward, expr=self.geometry.src_p_x * dt)
        src_p_z_term = self.geometry.src_p_z.inject(field=self.sigma[1, 1].forward, expr=self.geometry.src_p_z * dt)
        return src_p_x_term + src_p_z_term

    def _build_receiver_terms(self) -> Sequence[object]:
        terms = []

        # New default data for FWI: displacement receivers.
        if self.geometry.rec_ux is not None:
            terms += self.geometry.rec_ux.interpolate(expr=self.u[0].forward)
        if self.geometry.rec_uz is not None:
            terms += self.geometry.rec_uz.interpolate(expr=self.u[1].forward)

        # Backward-compatible optional velocity receivers.  They are only built
        # when Geometry(receiver_components=("vx", "vz", ...)) is requested.
        if self.geometry.rec_vx is not None:
            terms += self.geometry.rec_vx.interpolate(expr=self.v[0].forward)
        if self.geometry.rec_vz is not None:
            terms += self.geometry.rec_vz.interpolate(expr=self.v[1].forward)

        if self.geometry.rec_p is not None:
            terms += self.geometry.rec_p.interpolate(expr=(self.sigma[0, 0].forward + self.sigma[1, 1].forward) / 2.0)
        return terms

    def _build_operator(self) -> Operator:
        d = self._build_staggered_derivatives()
        equations = []
        equations += self._build_velocity_update(d)
        equations += self._build_stress_to_velocity_cpml(d)
        equations += self._build_displacement_update()
        equations += self._build_displacement_save()
        equations += self._build_stress_update(d)
        equations += self._build_velocity_to_stress_cpml(d)
        equations += self._build_source_terms()
        equations += self._build_receiver_terms()

        return Operator(
            equations,
            subs=self.grid.spacing_map,
            opt=self.devito_opt,
        )

    # ------------------------------------------------------------------
    # Runtime API
    # ------------------------------------------------------------------

    def state_fields(self) -> Sequence[Function]:
        fields = [
            self.v[0], self.v[1],
            self.u[0], self.u[1],
            self.sigma[0, 0], self.sigma[0, 1], self.sigma[1, 0], self.sigma[1, 1],
            *self.pml_memory_fields,
        ]
        if self.u_save is not None:
            fields += [self.u_save[0], self.u_save[1]]
        return fields

    def reset(self) -> None:
        """Reset wavefields, CPML memories, and receiver buffers."""
        for func in self.state_fields():
            func.data[:] = 0.0
        for rec in self.geometry.receivers.values():
            rec.data[:] = 0.0

    def forward(
        self,
        *,
        reset: bool = True,
        time: Optional[int] = None,
        dt: Optional[float] = None,
        return_wavefields: bool = False,
        return_u_save: bool = False,
        notebook_scaling: bool = False,
        f_norm_x: float = 1.0e10,
        f_norm_z: float = 1.0e10,
        f_norm_p: float = 1.0e5,
        **operator_kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Run forward modeling and return receiver data.

        By default outputs are raw Devito receiver arrays.  The default geometry
        records displacement components ``ux``/``uz`` instead of velocity.  Set
        save_u=True at solver construction time and return_u_save=True here to
        return sparsely saved full displacement wavefields.

        notebook_scaling=True is retained only for backward-compatible velocity
        and pressure diagnostics: vx*=1e10, vz*=1e10, p*=-1e5.
        """
        if reset:
            self.reset()

        run_time = self.geometry.nt - 1 if time is None else int(time)
        run_dt = self.geometry.dt if dt is None else float(dt)
        self.operator(time=run_time, dt=run_dt, **operator_kwargs)

        out: Dict[str, np.ndarray] = {}
        for comp, rec in self.geometry.receivers.items():
            data = np.array(rec.data, copy=True)
            if notebook_scaling:
                if comp == "vx":
                    data = data * f_norm_x
                elif comp == "vz":
                    data = data * f_norm_z
                elif comp == "p":
                    data = -data * f_norm_p
            out[comp] = data

        if return_wavefields:
            out["u_x"] = np.array(self.u[0].data, copy=True)
            out["u_z"] = np.array(self.u[1].data, copy=True)
            out["v_x"] = np.array(self.v[0].data, copy=True)
            out["v_z"] = np.array(self.v[1].data, copy=True)
            out["sigma_xx"] = np.array(self.sigma[0, 0].data, copy=True)
            out["sigma_xz"] = np.array(self.sigma[0, 1].data, copy=True)
            out["sigma_zz"] = np.array(self.sigma[1, 1].data, copy=True)

        if return_u_save:
            if self.u_save is None:
                raise ValueError("return_u_save=True requires ElasticCPMLSolver(..., save_u=True).")
            out["u_save_x"] = np.array(self.u_save[0].data, copy=True)
            out["u_save_z"] = np.array(self.u_save[1].data, copy=True)
            out["u_save_interval"] = np.array(self.u_save_interval)

        return out


# -----------------------------------------------------------------------------
# Gradient map / adjoint-state gradient
# -----------------------------------------------------------------------------


class ElasticCPMLGradientMap(ElasticCPMLSolver):
    """
    Build lambda/mu gradients with a reversed-time adjoint-state calculation.

    This class deliberately reuses the same Model/Geometry objects and the same
    handwritten CPML/staggered 4th-order update style as ``ElasticCPMLSolver``.

    Implementation convention
    -------------------------
    Instead of constructing a Devito backward-time operator, the residual data
    are reversed in NumPy and the adjoint wavefield is propagated forward in a
    new time variable tau.  Therefore

        tau = 0, 1, ..., nt-1   corresponds to   forward time nt-1, ..., 0.

    The saved forward displacement wavefield is also reversed:

        u_fwd_rev[j] = u_save[n_save - 1 - j].

    For this first implementation we require

        (geometry.nt - 1) % u_save_interval == 0

    so that the reversed coarse-time forward wavefield aligns exactly with the
    adjoint coarse time.

    Notes
    -----
    The CPML used here is a practical absorbing CPML for the reversed-time
    adjoint propagation.  It is not yet the fully transposed discrete adjoint of
    every CPML auxiliary equation.  This is usually the right first step for FWI
    development; a strict adjoint test can be added later if needed.
    """

    def __init__(
        self,
        model: ElasticModel,
        geometry: AcquisitionGeometry,
        residual_ux: np.ndarray,
        residual_uz: np.ndarray,
        u_save_x: np.ndarray,
        u_save_z: np.ndarray,
        u_save_interval: int,
        *,
        space_order: int = 4,
        time_order: int = 1,
        dtype: object = np.float32,
        cfl: Optional[float] = 0.4,
        gradient_scale: Optional[float] = None,
        adjoint_source_scale: float = 1.0,
        devito_opt: object = ("advanced", {"cire-mingain": 50}),
        configure_devito: bool = True,
    ) -> None:
        self.model = model
        self.geometry = geometry
        self.residual_ux = np.asarray(residual_ux)
        self.residual_uz = np.asarray(residual_uz)
        self.input_u_save_x = np.asarray(u_save_x)
        self.input_u_save_z = np.asarray(u_save_z)
        self.u_save_interval = int(u_save_interval)
        self.space_order = int(space_order)
        self.time_order = int(time_order)
        self.dtype = _as_dtype(dtype)
        self.cfl = cfl
        self.gradient_scale = float(self.u_save_interval if gradient_scale is None else gradient_scale)
        self.adjoint_source_scale = float(adjoint_source_scale)
        self.devito_opt = devito_opt
        self.configure_devito = configure_devito

        if self.space_order != 4:
            raise NotImplementedError("GradientMap currently preserves only the forward solver's space_order=4 stencil.")
        if self.time_order != 1:
            raise NotImplementedError("GradientMap currently preserves only the forward solver's time_order=1 update.")
        if self.model.grid is not self.geometry.model.grid:
            raise ValueError("Geometry must be built from the same ElasticModel instance as GradientMap.")
        if self.cfl is not None:
            self.model.check_time_step(self.geometry.dt, cfl=float(self.cfl))
        if self.configure_devito:
            configuration["opt"] = "advanced"
            configuration["profiling"] = "advanced"

        self.grid = self.model.grid
        self.x, self.z = self.grid.dimensions
        self.t = self.grid.stepping_dim
        self.dt_symbol = self.grid.time_dim.spacing

        self.px = Dimension(name="px_g")
        self.pz = Dimension(name="pz_g")
        self.ps = Dimension(name="ps_g")
        self.LEFT = self.BOTTOM = 0
        self.RIGHT = self.TOP = 1
        self.x_left = self.px
        self.x_right = self.x.symbolic_max - self.px
        self.z_bottom = self.pz
        self.z_top = self.z.symbolic_max - self.pz

        self._validate_and_prepare_reversed_inputs()
        self._build_fields()
        self._build_model_parameter_functions()
        self._build_pml_memory_fields()
        self._build_pml_coefficient_functions()
        self._build_adjoint_sources()
        self._build_gradient_functions()
        self.operator = self._build_operator()
        self.reset()

    # ------------------------------------------------------------------
    # Input preparation and validation
    # ------------------------------------------------------------------

    def _validate_and_prepare_reversed_inputs(self) -> None:
        if self.u_save_interval <= 0:
            raise ValueError("u_save_interval must be positive.")

        nt = self.geometry.nt
        nrec = self.geometry.rec_coordinates.get("ux", next(iter(self.geometry.rec_coordinates.values()))).shape[0]

        if self.residual_ux.shape != (nt, nrec):
            raise ValueError(f"residual_ux must have shape {(nt, nrec)}, got {self.residual_ux.shape}.")
        if self.residual_uz.shape != (nt, nrec):
            raise ValueError(f"residual_uz must have shape {(nt, nrec)}, got {self.residual_uz.shape}.")

        if (nt - 1) % self.u_save_interval != 0:
            raise ValueError(
                "GradientMap currently requires (geometry.nt - 1) % u_save_interval == 0 "
                "so that residual[::-1] and u_save[::-1] align exactly in time. "
                f"Got nt={nt}, u_save_interval={self.u_save_interval}."
            )

        self.n_u_save = 1 + (nt - 1) // self.u_save_interval
        expected_save_shape = (self.n_u_save, *self.model.extended_shape)
        if self.input_u_save_x.shape != expected_save_shape:
            raise ValueError(f"u_save_x must have shape {expected_save_shape}, got {self.input_u_save_x.shape}.")
        if self.input_u_save_z.shape != expected_save_shape:
            raise ValueError(f"u_save_z must have shape {expected_save_shape}, got {self.input_u_save_z.shape}.")

        # Reverse time outside Devito.  The adjoint operator then propagates
        # forward in tau, while physically moving backward in the original time.
        self.residual_ux_rev = np.ascontiguousarray(self.residual_ux[::-1], dtype=self.dtype)
        self.residual_uz_rev = np.ascontiguousarray(self.residual_uz[::-1], dtype=self.dtype)
        self.u_save_x_rev = np.ascontiguousarray(self.input_u_save_x[::-1], dtype=self.dtype)
        self.u_save_z_rev = np.ascontiguousarray(self.input_u_save_z[::-1], dtype=self.dtype)

        if "ux" in self.geometry.rec_coordinates:
            self.adjoint_coordinates = self.geometry.rec_coordinates["ux"]
        else:
            self.adjoint_coordinates = next(iter(self.geometry.rec_coordinates.values()))
        self.adjoint_coordinates = np.ascontiguousarray(self.adjoint_coordinates, dtype=np.float64)

    # ------------------------------------------------------------------
    # Field construction
    # ------------------------------------------------------------------

    def _build_fields(self) -> None:
        # The inherited CPML and update methods operate on self.v, self.u, and
        # self.sigma.  In this class they represent adjoint velocity,
        # adjoint displacement, and adjoint stress-like fields.
        self.v = VectorTimeFunction(
            name="v_adj",
            grid=self.grid,
            time_order=self.time_order,
            space_order=self.space_order,
            dtype=self.dtype,
        )
        self.u = VectorTimeFunction(
            name="u_adj",
            grid=self.grid,
            time_order=self.time_order,
            space_order=self.space_order,
            dtype=self.dtype,
        )
        self.sigma = TensorTimeFunction(
            name="sigma_adj",
            grid=self.grid,
            time_order=self.time_order,
            space_order=self.space_order,
            dtype=self.dtype,
        )

        self.coarse_time = ConditionalDimension(
            name="coarse_t_g",
            parent=self.grid.time_dim,
            factor=self.u_save_interval,
        )
        self.u_fwd_rev = VectorTimeFunction(
            name="u_fwd_rev",
            grid=self.grid,
            time_order=0,
            space_order=self.space_order,
            save=self.n_u_save,
            time_dim=self.coarse_time,
            dtype=self.dtype,
        )
        self.u_fwd_rev[0].data[:] = self.u_save_x_rev
        self.u_fwd_rev[1].data[:] = self.u_save_z_rev

        # Keep these attributes for compatibility with inherited helpers.
        self.u_save = None

    def _build_adjoint_sources(self) -> None:
        kwargs: Dict[str, object] = dict(
            grid=self.grid,
            npoint=self.adjoint_coordinates.shape[0],
            time_range=self.geometry.time_axis,
            dtype=self.dtype,
        )
        if self.geometry.interpolation is not None:
            kwargs["interpolation"] = self.geometry.interpolation

        self.adj_src_ux = PointSource(name="adj_src_ux", **kwargs)
        self.adj_src_uz = PointSource(name="adj_src_uz", **kwargs)
        self.adj_src_ux.coordinates.data[:, :] = self.adjoint_coordinates
        self.adj_src_uz.coordinates.data[:, :] = self.adjoint_coordinates
        self.adj_src_ux.data[:] = self.residual_ux_rev
        self.adj_src_uz.data[:] = self.residual_uz_rev

    def _build_gradient_functions(self) -> None:
        # The gradient containers themselves do not need space_order.  The
        # derivative expressions that feed them are handwritten below using the
        # same 4th-order staggered stencil as the forward stress update.
        self.grad_lam = Function(name="grad_lam", grid=self.grid, dtype=self.dtype)
        self.grad_mu = Function(name="grad_mu", grid=self.grid, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Overrides: sources, receivers, saving, gradient
    # ------------------------------------------------------------------

    def _build_source_terms(self) -> Sequence[object]:
        # Residuals are displacement residuals, but in this first-order system
        # they are injected as adjoint forcing into v_adj.forward before
        # u_adj.forward is integrated.  The scale is deliberately explicit so
        # that dot-product/Taylor tests can tune sign and units if needed.
        dt = self.dt_symbol
        scale = self.dtype(self.adjoint_source_scale)
        return (
            self.adj_src_ux.inject(field=self.v[0].forward, expr=scale * dt * self.adj_src_ux)
            + self.adj_src_uz.inject(field=self.v[1].forward, expr=scale * dt * self.adj_src_uz)
        )

    def _build_receiver_terms(self) -> Sequence[object]:
        return []

    def _build_displacement_save(self) -> Sequence[object]:
        return []

    def _strain_derivatives(self, vec: VectorTimeFunction, time_index) -> Dict[str, object]:
        """Handwritten 4th-order staggered derivatives for a vector field.

        This mirrors the forward solver's velocity-to-stress derivative block:

            dvx_dx, dvz_dz, dvx_dz, dvz_dx

        but is used here for displacement-like fields.  ``time_index`` is either
        ``self.t + 1`` for the current adjoint displacement or ``self.coarse_time``
        for the reversed saved forward displacement.
        """
        x, z = self.x, self.z
        hx = x.spacing
        hz = z.spacing
        c1 = self.dtype(9.0 / 8.0)
        c2 = self.dtype(1.0 / 24.0)

        qx = vec[0]
        qz = vec[1]

        dqx_dx = (c1 * (qx[time_index, x, z] - qx[time_index, x - 1, z])
                  - c2 * (qx[time_index, x + 1, z] - qx[time_index, x - 2, z])) / hx
        dqz_dz = (c1 * (qz[time_index, x, z] - qz[time_index, x, z - 1])
                  - c2 * (qz[time_index, x, z + 1] - qz[time_index, x, z - 2])) / hz
        dqx_dz = (c1 * (qx[time_index, x, z + 1] - qx[time_index, x, z])
                  - c2 * (qx[time_index, x, z + 2] - qx[time_index, x, z - 1])) / hz
        dqz_dx = (c1 * (qz[time_index, x + 1, z] - qz[time_index, x, z])
                  - c2 * (qz[time_index, x + 2, z] - qz[time_index, x - 1, z])) / hx

        return {
            "dqx_dx": dqx_dx,
            "dqz_dz": dqz_dz,
            "dqx_dz": dqx_dz,
            "dqz_dx": dqz_dx,
        }

    def _build_gradient_update(self) -> Sequence[object]:
        # u_adj.forward is the newly updated adjoint displacement at the current
        # reversed-time step.  u_fwd_rev[coarse_t] is the matching saved forward
        # displacement frame because both residual and u_save were reversed.
        p = self._strain_derivatives(self.u, self.t + 1)
        u = self._strain_derivatives(self.u_fwd_rev, self.coarse_time)

        div_p = p["dqx_dx"] + p["dqz_dz"]
        div_u = u["dqx_dx"] + u["dqz_dz"]

        two = self.dtype(2.0)
        half = self.dtype(0.5)
        scale = self.dtype(self.gradient_scale)

        sym_p_xx = two * p["dqx_dx"]
        sym_p_zz = two * p["dqz_dz"]
        sym_p_xz = p["dqx_dz"] + p["dqz_dx"]

        sym_u_xx = two * u["dqx_dx"]
        sym_u_zz = two * u["dqz_dz"]
        sym_u_xz = u["dqx_dz"] + u["dqz_dx"]

        inner = sym_u_xx * sym_p_xx + two * sym_u_xz * sym_p_xz + sym_u_zz * sym_p_zz

        return [
            Inc(self.grad_lam, scale * div_u * div_p, implicit_dims=[self.coarse_time]),
            Inc(self.grad_mu, half * scale * inner, implicit_dims=[self.coarse_time]),
        ]

    def _build_operator(self) -> Operator:
        d = self._build_staggered_derivatives()
        equations = []
        equations += self._build_velocity_update(d)
        equations += self._build_stress_to_velocity_cpml(d)
        equations += self._build_source_terms()
        equations += self._build_displacement_update()
        equations += self._build_stress_update(d)
        equations += self._build_velocity_to_stress_cpml(d)
        equations += self._build_gradient_update()

        return Operator(
            equations,
            subs=self.grid.spacing_map,
            opt=self.devito_opt,
        )

    # ------------------------------------------------------------------
    # Runtime API
    # ------------------------------------------------------------------

    def state_fields(self) -> Sequence[Function]:
        return [
            self.v[0], self.v[1],
            self.u[0], self.u[1],
            self.sigma[0, 0], self.sigma[0, 1], self.sigma[1, 0], self.sigma[1, 1],
            *self.pml_memory_fields,
            self.grad_lam, self.grad_mu,
        ]

    def reset(self) -> None:
        for func in self.state_fields():
            func.data[:] = 0.0
        # Do not reset u_fwd_rev or adjoint source data; those are fixed inputs.

    def apply(
        self,
        *,
        reset: bool = True,
        time: Optional[int] = None,
        dt: Optional[float] = None,
        physical_domain_only: bool = True,
        return_adjoint_wavefield: bool = False,
        **operator_kwargs,
    ) -> Dict[str, np.ndarray]:
        if reset:
            self.reset()

        run_time = self.geometry.nt - 1 if time is None else int(time)
        run_dt = self.geometry.dt if dt is None else float(dt)
        self.operator(time=run_time, dt=run_dt, **operator_kwargs)

        grad_lam = np.array(self.grad_lam.data, copy=True)
        grad_mu = np.array(self.grad_mu.data, copy=True)

        if physical_domain_only and self.model.pml.npml > 0:
            n = self.model.pml.npml
            grad_lam = grad_lam[n:-n, n:-n]
            grad_mu = grad_mu[n:-n, n:-n]

        out: Dict[str, np.ndarray] = {
            "grad_lam": grad_lam,
            "grad_mu": grad_mu,
        }

        if return_adjoint_wavefield:
            out["u_adj_x"] = np.array(self.u[0].data, copy=True)
            out["u_adj_z"] = np.array(self.u[1].data, copy=True)
            out["v_adj_x"] = np.array(self.v[0].data, copy=True)
            out["v_adj_z"] = np.array(self.v[1].data, copy=True)

        return out


# -----------------------------------------------------------------------------
# High-level FWI workflow helpers: model I/O and param-dict builders
# -----------------------------------------------------------------------------


def _param_get(param: Mapping[str, object], key: str, default=None):
    """Read a key from a possibly nested FWI parameter dictionary."""
    if param is None:
        return default
    if key in param:
        return param[key]
    # Common nested groups used by scripts/notebooks.
    for group in ("model", "geometry", "solver", "pml", "io"):
        value = param.get(group) if isinstance(param, Mapping) else None
        if isinstance(value, Mapping) and key in value:
            return value[key]
    return default


def build_pml_config(param: Mapping[str, object]) -> PMLConfig:
    defaults = PMLConfig()
    pml_param = param.get("pml", {}) if isinstance(param, Mapping) else {}
    if pml_param is None:
        pml_param = {}
    if not isinstance(pml_param, Mapping):
        raise TypeError("param['pml'] must be a mapping when provided.")

    vpmax = pml_param.get(
        "vpmax",
        pml_param.get("vp_max", _param_get(param, "vpmax", _param_get(param, "vp_max", None))),
    )

    return PMLConfig(
        npml=int(pml_param.get("npml", _param_get(param, "npml", defaults.npml))),
        r_val=float(pml_param.get("r_val", _param_get(param, "r_val", defaults.r_val))),
        n_power=int(pml_param.get("n_power", _param_get(param, "n_power", defaults.n_power))),
        tol=float(pml_param.get("tol", _param_get(param, "tol", defaults.tol))),
        alpha0=pml_param.get("alpha0", _param_get(param, "alpha0", defaults.alpha0)),
        vpmax=None if vpmax is None else float(vpmax),
    )


def _load_npy_if_exists(path: Path) -> Optional[np.ndarray]:
    return np.load(path) if path.exists() else None


def load_model(
    model_path: Union[str, Path],
    param: Mapping[str, object],
    *,
    vp_name: str = "vp.npy",
    vs_name: str = "vs.npy",
    rho_name: str = "rho.npy",
    dtype: object = None,
) -> ElasticModel:
    """Load ``vp.npy``/``vs.npy`` from disk and build an :class:`ElasticModel`.

    Parameters
    ----------
    model_path
        Directory containing at least ``vp.npy`` and ``vs.npy``.
    param
        Dictionary containing model/grid/PML information.  Required entries are
        ``spacing``.  Optional entries include ``origin``, ``rho``, ``dtype`` and
        any PML parameters such as ``npml`` or nested ``pml``.
    vp_name, vs_name, rho_name
        File names inside ``model_path``.  ``rho.npy`` is optional; if absent,
        ``param['rho']`` is used.  If that is absent too, density defaults to 1.
    dtype
        Optional override for numerical dtype.  Defaults to ``param['dtype']``
        and then ``np.float32``.

    Notes
    -----
    The stored arrays are interpreted as physical-domain arrays.  PML padding is
    handled by :class:`ElasticModel` exactly as in the forward/gradient solvers.
    """
    model_dir = Path(model_path).expanduser()
    if not model_dir.exists():
        raise FileNotFoundError(f"model_path does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"model_path must be a directory: {model_dir}")

    vp_file = model_dir / vp_name
    vs_file = model_dir / vs_name
    if not vp_file.exists():
        raise FileNotFoundError(f"Missing vp file: {vp_file}")
    if not vs_file.exists():
        raise FileNotFoundError(f"Missing vs file: {vs_file}")

    dtype = _as_dtype(dtype if dtype is not None else _param_get(param, "dtype", np.float32))
    vp = np.ascontiguousarray(np.load(vp_file), dtype=dtype)
    vs = np.ascontiguousarray(np.load(vs_file), dtype=dtype)
    if vp.shape != vs.shape:
        raise ValueError(f"vp and vs must have the same shape, got {vp.shape} and {vs.shape}.")
    if vp.ndim != 2:
        raise ValueError(f"vp/vs must be 2-D physical-domain arrays, got ndim={vp.ndim}.")

    rho_arr = _load_npy_if_exists(model_dir / rho_name)
    if rho_arr is None:
        rho_value = _param_get(param, "rho", _param_get(param, "rho0", 1.0))
        rho = rho_value
    else:
        rho = np.ascontiguousarray(rho_arr, dtype=dtype)
        if rho.shape != vp.shape:
            raise ValueError(f"rho must have shape {vp.shape}, got {rho.shape}.")

    spacing = tuple(_param_get(param, "spacing"))
    origin = tuple(_param_get(param, "origin", (0.0, 0.0)))
    pml = build_pml_config(param)

    return ElasticModel(
        vp=vp,
        vs=vs,
        rho=rho,
        shape=vp.shape,
        spacing=spacing,
        origin=origin,
        pml=pml,
        dtype=dtype,
    )


def update_model(
    model: ElasticModel,
    save_path: Union[str, Path],
    *,
    vp: Optional[np.ndarray] = None,
    vs: Optional[np.ndarray] = None,
    rho: Optional[Union[float, np.ndarray]] = None,
    save_rho: bool = False,
    save_metadata: bool = True,
    overwrite: bool = True,
) -> Dict[str, Path]:
    """Save the current or updated physical model arrays to ``save_path``.

    This helper is intended for FWI iterations.  It writes

    ``save_path/vp.npy`` and ``save_path/vs.npy``

    and, optionally, ``rho.npy`` plus a small ``model_metadata.json`` file.

    Parameters
    ----------
    model
        Existing :class:`ElasticModel` that defines shape, spacing, origin, PML
        settings and default arrays.
    save_path
        Output directory.  It is created if needed.
    vp, vs, rho
        Optional updated arrays.  If omitted, the arrays currently stored in
        ``model`` are saved.  Passing ``vp``/``vs`` does not mutate ``model``;
        it only controls what is written to disk.
    save_rho
        Whether to write ``rho.npy``.  For constant-density workflows, keeping
        density in ``param`` is often cleaner, so this defaults to ``False``.
    save_metadata
        Whether to write ``model_metadata.json`` with grid/PML information.
    overwrite
        If ``False``, raise if target files already exist.

    Returns
    -------
    dict
        Mapping from logical names to written paths.
    """
    out_dir = Path(save_path).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = model.dtype
    vp_out = _as_physical_array(model.vp_array if vp is None else vp, model.shape, dtype)
    vs_out = _as_physical_array(model.vs_array if vs is None else vs, model.shape, dtype)
    rho_out = _as_physical_array(model.rho_array if rho is None else rho, model.shape, dtype)

    paths = {
        "vp": out_dir / "vp.npy",
        "vs": out_dir / "vs.npy",
    }
    if save_rho:
        paths["rho"] = out_dir / "rho.npy"
    if save_metadata:
        paths["metadata"] = out_dir / "model_metadata.json"

    if not overwrite:
        existing = [str(p) for p in paths.values() if p.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite existing files: " + ", ".join(existing))

    np.save(paths["vp"], np.ascontiguousarray(vp_out, dtype=dtype))
    np.save(paths["vs"], np.ascontiguousarray(vs_out, dtype=dtype))
    if save_rho:
        np.save(paths["rho"], np.ascontiguousarray(rho_out, dtype=dtype))

    if save_metadata:
        metadata = {
            "shape": list(model.shape),
            "spacing": list(model.spacing),
            "origin": list(model.origin),
            "dtype": np.dtype(dtype).name,
            "pml": {
                "npml": model.pml.npml,
                "r_val": model.pml.r_val,
                "n_power": model.pml.n_power,
                "tol": model.pml.tol,
                "alpha0": model.pml.alpha0,
                "vpmax": model.pml.vpmax,
            },
            "files": {
                "vp": "vp.npy",
                "vs": "vs.npy",
                "rho": "rho.npy" if save_rho else None,
            },
        }
        paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return paths


def build_model_from_param(param: Mapping[str, object]) -> ElasticModel:
    """Build a model from a parameter dictionary.

    If ``param['model_path']`` is present, arrays are loaded from that directory.
    Otherwise ``vp`` and ``vs`` must be present directly in ``param``.
    """
    model_path = _param_get(param, "model_path", None)
    if model_path is not None:
        return load_model(model_path, param)

    dtype = _as_dtype(_param_get(param, "dtype", np.float32))
    vp = _param_get(param, "vp")
    vs = _param_get(param, "vs")
    if vp is None or vs is None:
        raise ValueError("build_model_from_param requires either model_path or direct vp/vs arrays.")
    vp_arr = np.asarray(vp, dtype=dtype)
    vs_arr = np.asarray(vs, dtype=dtype)
    if vp_arr.shape != vs_arr.shape or vp_arr.ndim != 2:
        raise ValueError(f"vp/vs must be 2-D arrays with the same shape, got {vp_arr.shape} and {vs_arr.shape}.")

    return ElasticModel(
        vp=vp_arr,
        vs=vs_arr,
        rho=_param_get(param, "rho", _param_get(param, "rho0", 1.0)),
        shape=vp_arr.shape,
        spacing=tuple(_param_get(param, "spacing")),
        origin=tuple(_param_get(param, "origin", (0.0, 0.0))),
        pml=build_pml_config(param),
        dtype=dtype,
    )


def build_geometry_from_param(model: ElasticModel, param: Mapping[str, object]) -> AcquisitionGeometry:
    """Build :class:`AcquisitionGeometry` from a flat or nested parameter dict."""
    dtype = _as_dtype(_param_get(param, "dtype", model.dtype))
    return AcquisitionGeometry(
        model=model,
        src_positions=_param_get(param, "src_positions"),
        rec_positions=_param_get(param, "rec_positions"),
        t0=float(_param_get(param, "t0", 0.0)),
        tn=float(_param_get(param, "tn")),
        dt=float(_param_get(param, "dt")),
        f0=float(_param_get(param, "f0")),
        dtype=dtype,
        interpolation=_param_get(param, "interpolation", None),
        receiver_components=tuple(_param_get(param, "receiver_components", ("ux", "uz", "p"))),
    )


def build_forward_solver_from_param(
    model: ElasticModel,
    geometry: AcquisitionGeometry,
    param: Mapping[str, object],
) -> ElasticCPMLSolver:
    """Build :class:`ElasticCPMLSolver` from ``model + geometry + param``."""
    return ElasticCPMLSolver(
        model=model,
        geometry=geometry,
        space_order=int(_param_get(param, "space_order", 4)),
        time_order=int(_param_get(param, "time_order", 1)),
        dtype=_as_dtype(_param_get(param, "dtype", model.dtype)),
        cfl=_param_get(param, "cfl", 0.4),
        save_u=bool(_param_get(param, "save_u", False)),
        u_save_interval=int(_param_get(param, "u_save_interval", 10)),
        devito_opt=_param_get(param, "devito_opt", ("advanced", {"cire-mingain": 50})),
        configure_devito=bool(_param_get(param, "configure_devito", True)),
    )


def build_gradient_map_from_param(
    model,
    geometry,
    residual_ux,
    residual_uz,
    u_save,
    param,
):
    """
    Build ElasticCPMLGradientMap from model, geometry, residual data, saved
    forward displacement wavefield, and a parameter dictionary.

    Parameters
    ----------
    model : ElasticModel
        Current model used for the synthetic forward modeling.

    geometry : AcquisitionGeometry
        Single-shot geometry. For adjoint-state gradient, receiver positions in
        this geometry become adjoint source positions.

    residual_ux, residual_uz : np.ndarray
        Displacement residuals for one shot, usually

            residual_ux = synthetic_ux - observed_ux
            residual_uz = synthetic_uz - observed_uz

        with shape = (nt, nrec).

    u_save : dict, tuple, list, or np.ndarray
        Saved forward displacement wavefield.

        Recommended input is the forward solver output dictionary:

            u_save = out

        where

            out["u_save_x"]
            out["u_save_z"]
            out["u_save_interval"]

        exist.

        Also accepted:

            u_save = {"u_save_x": ..., "u_save_z": ...}
            u_save = (u_save_x, u_save_z)

    param : dict
        Parameter dictionary. It may be flat or nested. Recommended nested form:

            param = {
                "solver": {
                    "space_order": 4,
                    "time_order": 1,
                    "dtype": np.float32,
                    "cfl": 0.4,
                    "u_save_interval": 10,
                    "configure_devito": False,
                },
                "gradient": {
                    "gradient_scale": None,
                    "adjoint_source_scale": 1.0,
                    "physical_domain_only": True,
                }
            }

    Returns
    -------
    ElasticCPMLGradientMap
        Constructed gradient-map solver. Run it with

            grad = grad_solver.apply()
    """

    # ------------------------------------------------------------------
    # Local parameter getter.
    #
    # If elastic_cpml_solver_fwi_io.py already has _param_get, you may replace
    # this local helper with calls to that global helper. Keeping it here makes
    # the function self-contained and safer to paste.
    # ------------------------------------------------------------------
    def _get_param(param, key, default=None, groups=("gradient", "solver", "pml", "geometry", "model")):
        if key in param:
            return param[key]
        for group in groups:
            block = param.get(group, None)
            if isinstance(block, dict) and key in block:
                return block[key]
        return default

    # ------------------------------------------------------------------
    # Extract saved forward wavefield.
    # ------------------------------------------------------------------
    if isinstance(u_save, dict):
        if "u_save_x" not in u_save or "u_save_z" not in u_save:
            raise ValueError(
                "u_save is a dict, but it must contain 'u_save_x' and 'u_save_z'. "
                "Usually pass the full forward output: out = solver.forward(return_u_save=True)."
            )

        u_save_x = u_save["u_save_x"]
        u_save_z = u_save["u_save_z"]

        # Prefer the actual interval returned by forward solver if available.
        if "u_save_interval" in u_save:
            u_save_interval = int(np.asarray(u_save["u_save_interval"]).item())
        else:
            u_save_interval = int(_get_param(param, "u_save_interval", 10))

    elif isinstance(u_save, (tuple, list)) and len(u_save) == 2:
        u_save_x, u_save_z = u_save
        u_save_interval = int(_get_param(param, "u_save_interval", 10))

    else:
        raise TypeError(
            "u_save must be either a dict with keys 'u_save_x'/'u_save_z', "
            "or a tuple/list like (u_save_x, u_save_z)."
        )

    # ------------------------------------------------------------------
    # Solver / gradient options.
    # ------------------------------------------------------------------
    space_order = int(_get_param(param, "space_order", 4))
    time_order = int(_get_param(param, "time_order", 1))
    dtype = _get_param(param, "dtype", np.float32)
    cfl = _get_param(param, "cfl", 0.4)

    gradient_scale = _get_param(param, "gradient_scale", None)
    adjoint_source_scale = float(_get_param(param, "adjoint_source_scale", 1.0))

    devito_opt = _get_param(
        param,
        "devito_opt",
        ("advanced", {"cire-mingain": 50}),
    )
    configure_devito = bool(_get_param(param, "configure_devito", True))

    # ------------------------------------------------------------------
    # Build GradientMap.
    # ------------------------------------------------------------------
    grad_solver = ElasticCPMLGradientMap(
        model=model,
        geometry=geometry,
        residual_ux=np.asarray(residual_ux),
        residual_uz=np.asarray(residual_uz),
        u_save_x=np.asarray(u_save_x),
        u_save_z=np.asarray(u_save_z),
        u_save_interval=u_save_interval,
        space_order=space_order,
        time_order=time_order,
        dtype=dtype,
        cfl=cfl,
        gradient_scale=gradient_scale,
        adjoint_source_scale=adjoint_source_scale,
        devito_opt=devito_opt,
        configure_devito=configure_devito,
    )

    return grad_solver


__all__ = [
    "PMLConfig",
    "ElasticModel",
    "AcquisitionGeometry",
    "ElasticCPMLSolver",
    "ElasticCPMLGradientMap",
    "build_pml_config",
    "load_model",
    "update_model",
    "build_model_from_param",
    "build_geometry_from_param",
    "build_forward_solver_from_param",
    "build_gradient_map_from_param",
    "pml_fraction_1d",
    "make_pml_ab_1d",
]