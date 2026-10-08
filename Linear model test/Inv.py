import numpy as np
import logging, os, sys

# Keep the BLAS/OpenMP thread count low because Dask already parallelizes over shots.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from devito import configuration
configuration["log-level"] = "WARNING"

import matplotlib.pyplot as plt

from Elastic_pml import *
from examples.seismic import Model, plot_velocity, plot_image

from dask.distributed import Client, LocalCluster, wait, progress
from devito import *

import time

if __name__ == '__main__':
    param = {
        "model":{
            'model_path': './Initial_models/Initial',
            'spacing': (10.,10.),
            'origin': (0.,0.)
        },
        
        "pml":{
            'npml': 20
        },
        
        "geometry":{
            'nshots': 40,
            'nrec': 250,
            'src_depth': 20,
            'rec_depth': 40,
            't0': 0.0,
            'tn': 3000.0,
            'dt': .5,
            'f0': 0.010,
            'receiver_components': ("ux", "uz"),
        },
        
        "solver":{
            'space_order': 4,
            'time_order': 1,
            'cfl': 0.4,
            'save_u': True,
            'u_save_interval': 10,
            'configure_devito': False,
        },
        
        "gradient": {
            "physical_domain_only": True,
            "adjoint_source_scale": -1.0,
            "gradient_scale": None,
            "return_dtype": "float64",
        },

        "data_path": './Data',
        "arrival_path": './Data_direct_arrival',
        "dask_nwrks": 10,
        "cutoff": './cutoff.npy',

        'vs_min': 1.,
        'vs_max': 3.,
        'var_min': 0.3,
        'var_max': 0.6,

        # Fallback for single-stage mode. The actual run below uses fwi_stages.
        'maxiter' : 60,
        'ftol': 1e-6,
        'maxls': 10,
        'maxcor': 10,

        'water_layer': 25,
        'nu': .1, # Used to control the lower bound of vp/vs
        "work_dir": "./Result",
        "save_dir": "./Intermediate",

        # Gradient mask/smoothing. The cutoff is always applied. Gaussian
        # smoothing is enabled only in the first stage below.
        'grad_smooth_sigma_x': 6,
        'grad_smooth_sigma_z': 3,

        # Preserve the time-step factor used in the original script. Set this
        # to 1.0 if Elastic_pml.apply() already includes the desired time
        # quadrature scaling in your current implementation.
        "gradient_time_scale": None,  # None -> geometry.dt * 1e-3

        # Two-stage inversion:
        #   stage 1: smooth gradient, mainly low-wavenumber update;
        #   stage 2: no smoothing, allowing higher-wavenumber update.
        # "fwi_stages": [
        #     {"name": "smooth",   "maxiter": 30, "smooth_gradient": True},
        #     {"name": "unsmooth", "maxiter": 30, "smooth_gradient": False},
        # ],

        "fwi_stages": [
            {"name": "smooth",   "maxiter": 60, "smooth_gradient": True},
        ],
        
        # Automatic numerical scaling. These two scales are initialized only at
        # the first loss evaluation and then kept fixed across both stages.
        # They are intentionally independent: objective scaling controls the
        # printed/line-search objective magnitude, while gradient scaling acts
        # as a scalar optimization preconditioner.
        "auto_scale": True,
        "objective_scale_target": 1e7,
        "gradient_scale_target": 1e-2,
        "scaling_eps": 1e-30,

        # Used only when auto_scale=False.
        "objective_scale": None,
        "gradient_post_scale": None,

        # Dask retry control. Recursive retry is deliberately avoided.
        "shot_max_retries": 0,
    }
    
    model = build_model_from_param(param)
    extent = (model.vp.shape[0]-1)*param['model']['spacing'][0]
    src_x = np.linspace(20., extent-20., param['geometry']['nshots'])
    rec_x = np.linspace(20., extent-20., param['geometry']['nrec'])
    param["geometry"]['src_positions'] = np.column_stack([
        src_x,
        np.full_like(src_x, param['geometry']['src_depth'])
    ])
    param["geometry"]['rec_positions'] = np.column_stack([
        rec_x,
        np.full_like(rec_x, param['geometry']['rec_depth'])
    ])
    
    print("Initializing local cluster...")
    cluster = LocalCluster(
        n_workers=param['dask_nwrks'],
        threads_per_worker=1,
        processes=True,
        silence_logs=logging.INFO,
        )
    client = Client(cluster)
    time.sleep(10)
    
    def inv_single_shot(param):
        max_retries = int(param.get("shot_max_retries", 0))
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                shot_id = param["shot_id"]
                model = build_model_from_param(param)
                geometry_s = build_geometry_from_param(model, param)
                solver = build_forward_solver_from_param(model, geometry_s, param)
                out = solver.forward(return_u_save=True)
                data = np.load(param['data_path'])
                
                # residual_ux = out['ux'] - data[0, :]
                # residual_uz = out['uz'] - data[1, :]
                
                # # Raw L2 objective. The previous hard-coded factor 1e2 is
                # # replaced by automatic scaling in fwi_loss().
                # objective = 0.5 * (
                #     np.sum(residual_ux.astype(np.float64) ** 2)
                #     + np.sum(residual_uz.astype(np.float64) ** 2)
                # )
                
                # from Wd1d_optimized import get_ot_derivative_equalize
            
                # data_arrival = np.load(param['arrival_path'])
                # x = np.arange(geometry_s.nt)*geometry_s.dt*1e-3
                # u_reduced = np.array([out['ux']] + [out['uz']]) - data_arrival
                # d_reduced = data - data_arrival
                
                # J1, residual_ux = get_ot_derivative_equalize(x, u_reduced[0,:], d_reduced[0,:], beta=5.)
                # J2, residual_uz = get_ot_derivative_equalize(x, u_reduced[1,:], d_reduced[1,:], beta=5.)
                
                # objective = J1+J2
                
                # from Wd1d_revised_sp_encoded_optimized import ot_dist_equalize_multitrace_with_grad_softplus_encoding
                from ROT_corrected import ot_dist_equalize_multitrace_with_grad_softplus_encoding
                data_arrival = np.load(param['arrival_path'])
                x = np.arange(geometry_s.nt)*geometry_s.dt*1e-3
                u_reduced = np.array([out['ux']] + [out['uz']]) - data_arrival
                d_reduced = data - data_arrival
                
                objective, residual = ot_dist_equalize_multitrace_with_grad_softplus_encoding(x, u_reduced, d_reduced, alpha=4e-3, beta=5.)
                
                residual_ux = residual[0,:]
                residual_uz = residual[1,:]
                
                grad_solver = build_gradient_map_from_param(
                    model=model,
                    geometry=geometry_s,
                    residual_ux=residual_ux,
                    residual_uz=residual_uz,
                    u_save=out,
                    param=param,
                )

                grad = grad_solver.apply(
                    physical_domain_only=param.get("gradient", {}).get("physical_domain_only", True)
                )

                # Preserve the dt*1e-3 factor used by the original script, but
                # make it explicit and configurable.
                gradient_time_scale = param.get("gradient_time_scale", None)
                if gradient_time_scale is None:
                    gradient_time_scale = geometry_s.dt * 1e-3
                gradient_time_scale = float(gradient_time_scale)

                grad_lam = np.asarray(grad["grad_lam"], dtype=np.float64) * gradient_time_scale
                grad_mu = np.asarray(grad["grad_mu"], dtype=np.float64) * gradient_time_scale

                return {
                    "shot_id": shot_id,
                    "objective": objective,
                    "grad_lam": grad_lam,
                    "grad_mu": grad_mu,
                }
            except Exception as e:
                last_error = e
                print(
                    f"Error computing gradient for shot {param.get('shot_id', 'unknown')} "
                    f"on attempt {attempt + 1}/{max_retries + 1}: {e}",
                    flush=True,
                )

        raise RuntimeError(
            f"Shot {param.get('shot_id', 'unknown')} failed after {max_retries + 1} attempts."
        ) from last_error
        

    def inv_multi_shots(param):
        import copy
        from scipy.ndimage import gaussian_filter

        nshots = param["geometry"]["nshots"]

        work = []
        for s in range(nshots):
            p = copy.deepcopy(param)

            p["shot_id"] = s

            # 每个 worker 只处理一个 source
            # 保持 shape=(1, 2)，不要传 shape=(2,)
            p["geometry"]["src_positions"] = np.asarray(
                param["geometry"]["src_positions"][s:s+1, :],
                dtype=np.float64,
            )

            # receiver 所有炮共用
            p["geometry"]["rec_positions"] = np.asarray(
                param["geometry"]["rec_positions"],
                dtype=np.float64,
            )

            # 每炮观测数据路径
            f0_mhz = int(param["geometry"]["f0"] * 1e3)
            obs_fname = f"Data_for_{s}_shot_{f0_mhz}hz.npy"
            p["data_path"] = os.path.join(param["data_path"], obs_fname)
            p["arrival_path"] = os.path.join(param["arrival_path"], obs_fname)

            work.append(p)

        futures = client.map(inv_single_shot, work)
        progress(futures)
        wait(futures)

        # gather 会把 worker 中的真实异常抛出来
        results = client.gather(futures)

        # 累加 objective 和梯度
        objective_total = 0.0
        grad_lam_total = None
        grad_mu_total = None

        for r in results:
            objective_total += float(r["objective"])

            gl = np.asarray(r["grad_lam"], dtype=np.float64)
            gm = np.asarray(r["grad_mu"], dtype=np.float64)

            if grad_lam_total is None:
                grad_lam_total = np.zeros_like(gl, dtype=np.float64)
                grad_mu_total = np.zeros_like(gm, dtype=np.float64)

            grad_lam_total += gl
            grad_mu_total += gm

        cutoff = np.load(param['cutoff']).astype(np.float64)
        smooth_gradient = bool(param.get("smooth_gradient", True))
        sigma_x = float(param.get("grad_smooth_sigma_x", 2.0))
        sigma_z = float(param.get("grad_smooth_sigma_z", 2.0))
        sigma = (sigma_x, sigma_z)

        # Always apply cutoff/mask. Gaussian smoothing is controlled by the
        # current FWI stage. For a taper cutoff, the smoothed version follows
        # the standard mask-smooth-mask pattern to avoid PML/water leakage.
        if smooth_gradient:
            grad_lam_total = cutoff * gaussian_filter(
                cutoff * grad_lam_total,
                sigma=sigma,
                mode="nearest",
            )
            grad_mu_total = cutoff * gaussian_filter(
                cutoff * grad_mu_total,
                sigma=sigma,
                mode="nearest",
            )
        else:
            grad_lam_total = cutoff * grad_lam_total
            grad_mu_total = cutoff * grad_mu_total

        # global is_first_call
        # if is_first_call:
        #     os.makedirs('./Grad', exist_ok=True)
        #     np.save('./Grad/f_10hz_raw.npy', objective_total)
        #     np.save('./Grad/gl_10hz_after_cutoff.npy', grad_lam_total)
        #     np.save('./Grad/gm_10hz_after_cutoff.npy', grad_mu_total)
        #     is_first_call = False
        
        return {
            "objective": objective_total,
            "grad_lam": grad_lam_total,
            "grad_mu": grad_mu_total,
        }
        
    def eta_from_nu(nu):
        return np.sqrt(2.0 * (1.0 - nu) / (1.0 - 2.0 * nu))
    
    
    def vpvs_to_x(vp, vs, param):
        """
        Convert full physical-domain vp/vs arrays to reduced optimization vector x.

        x = [vs_reduced, var_reduced]
        var = vp - eta * vs

        The first water_layer depth columns are excluded:
            vp[:, :water_layer], vs[:, :water_layer]
        """

        water = int(param["water_layer"])
        eta = eta_from_nu(param["nu"])

        vp = np.asarray(vp, dtype=np.float64)
        vs = np.asarray(vs, dtype=np.float64)

        vp_reduced = vp[:, water:]
        vs_reduced = vs[:, water:]
        var_reduced = vp_reduced - eta * vs_reduced

        return np.hstack([
            vs_reduced.ravel(),
            var_reduced.ravel(),
        ]).astype(np.float64)
        
    def x_to_vpvs(x, shape, param, vp_water, vs_water):
        """
        Convert reduced optimization vector x = [vs_reduced, var_reduced]
        back to full physical-domain vp/vs arrays.

        Model shape is (nx, nz). The first water_layer depth columns, i.e.
        [:, :water_layer], are fixed homogeneous water layer.

        x contains only the non-water part:
            vs[:, water_layer:]
            var[:, water_layer:]

        where
            var = vp - eta * vs
            vp  = var + eta * vs
        """

        water = int(param["water_layer"])
        eta = eta_from_nu(param["nu"])

        nx, nz = tuple(shape)
        n_reduced = nx * (nz - water)

        x = np.asarray(x, dtype=np.float64)
        if x.size != 2 * n_reduced:
            raise ValueError(
                f"x has wrong size. Expected {2 * n_reduced}, got {x.size}."
            )

        vs_reduced = x[:n_reduced].reshape(nx, nz - water)
        var_reduced = x[n_reduced:].reshape(nx, nz - water)

        vp_reduced = var_reduced + eta * vs_reduced

        vp = np.empty((nx, nz), dtype=np.float64)
        vs = np.empty((nx, nz), dtype=np.float64)

        # Fixed homogeneous water layer.
        vp[:, :water] = float(vp_water)
        vs[:, :water] = float(vs_water)

        # Inversion region.
        vp[:, water:] = vp_reduced
        vs[:, water:] = vs_reduced

        return vp.astype(np.float32), vs.astype(np.float32)
    
    
    def lame_grad_to_x_grad(grad_lam, grad_mu, vp, vs, rho, param):
        """
        Convert gradients w.r.t. lambda/mu to gradients w.r.t.
        optimization variables x = [vs_reduced, var_reduced],
        where var = vp - eta * vs.

        Assumes grad_lam/grad_mu are physical-domain arrays with shape=(nx, nz),
        same as vp/vs.
        """

        water = int(param["water_layer"])
        eta = eta_from_nu(param["nu"])

        grad_lam = np.asarray(grad_lam, dtype=np.float64)
        grad_mu = np.asarray(grad_mu, dtype=np.float64)
        vp = np.asarray(vp, dtype=np.float64)
        vs = np.asarray(vs, dtype=np.float64)
        rho = np.asarray(rho, dtype=np.float64)

        if rho.ndim == 0:
            rho = np.full_like(vp, float(rho), dtype=np.float64)

        grad_vp = 2.0 * rho * vp * grad_lam
        grad_vs = 2.0 * rho * vs * (grad_mu - 2.0 * grad_lam)

        # Chain rule for var = vp - eta * vs, so vp = var + eta * vs.
        grad_vs_param = grad_vs + eta * grad_vp
        grad_var = grad_vp

        grad_vs_reduced = grad_vs_param[:, water:]
        grad_var_reduced = grad_var[:, water:]

        grad_x = np.hstack([
            grad_vs_reduced.ravel(),
            grad_var_reduced.ravel(),
        ])

        # Return the raw gradient in optimization variables. The scalar
        # post-scale is applied centrally in fwi_loss(), so it can be
        # initialized automatically from the first evaluation and then locked.
        return grad_x.astype(np.float64)
    
    
    def build_fwi_bounds(x0, param):
        L = len(x0) // 2

        vs_min = float(param["vs_min"])
        vs_max = float(param["vs_max"])
        var_min = float(param.get("var_min", 0.0))
        var_max = float(param["var_max"])

        return [(vs_min, vs_max)] * L + [(var_min, var_max)] * L
    
    def get_fwi_stages(param):
        """
        Normalize the FWI stage configuration.

        Default behavior is a single stage using param['maxiter']. In this
        script, param['fwi_stages'] is set to two stages:
            1) smoothed gradient, 40 iterations;
            2) unsmoothed gradient, 40 iterations.
        """
        stages = param.get("fwi_stages", None)

        if stages is None:
            return [{
                "name": "single",
                "maxiter": int(param.get("maxiter", 100)),
                "smooth_gradient": bool(param.get("smooth_gradient", True)),
            }]

        normalized = []
        for i, stage in enumerate(stages, start=1):
            name = str(stage.get("name", f"stage_{i}"))
            maxiter = int(stage.get("maxiter", param.get("maxiter", 100)))
            if maxiter <= 0:
                raise ValueError(f"FWI stage {name!r} has non-positive maxiter={maxiter}.")

            normalized.append({
                "name": name,
                "maxiter": maxiter,
                "smooth_gradient": bool(stage.get("smooth_gradient", True)),
            })

        if len(normalized) == 0:
            raise ValueError("param['fwi_stages'] is empty.")

        return normalized


    def initialize_or_get_scales(objective_raw, grad_x_raw, param, state):
        """
        Initialize objective/gradient scales at the first loss evaluation.

        The scales are chosen so that at the first evaluation,
            scaled objective ~ objective_scale_target,
            scaled |grad_x|_inf ~ gradient_scale_target.

        After initialization, the same scales are reused in both FWI stages.
        """
        if state.get("scale_initialized", False):
            return float(state["objective_scale"]), float(state["gradient_post_scale"])

        eps = float(param.get("scaling_eps", 1e-30))
        auto_scale = bool(param.get("auto_scale", True))

        objective_raw_abs = abs(float(objective_raw))
        grad_norm_raw = float(np.max(np.abs(grad_x_raw)))

        if auto_scale:
            obj_target = float(param.get("objective_scale_target", 1e7))
            grad_target = float(param.get("gradient_scale_target", 1e-2))

            if np.isfinite(objective_raw_abs) and objective_raw_abs > eps:
                objective_scale = obj_target / objective_raw_abs
            else:
                objective_scale = 1.0
                print(
                    "[scaling warning] raw objective is zero/non-finite; "
                    "objective_scale is set to 1.0.",
                    flush=True,
                )

            if np.isfinite(grad_norm_raw) and grad_norm_raw > eps:
                gradient_post_scale = grad_target / grad_norm_raw
            else:
                gradient_post_scale = 1.0
                print(
                    "[scaling warning] raw grad norm is zero/non-finite; "
                    "gradient_post_scale is set to 1.0.",
                    flush=True,
                )
        else:
            objective_scale = param.get("objective_scale", 1.0)
            gradient_post_scale = param.get("gradient_post_scale", 1.0)
            objective_scale = 1.0 if objective_scale is None else float(objective_scale)
            gradient_post_scale = 1.0 if gradient_post_scale is None else float(gradient_post_scale)

        state["objective_scale"] = float(objective_scale)
        state["gradient_post_scale"] = float(gradient_post_scale)
        state["scale_initialized"] = True
        state["initial_raw_objective"] = float(objective_raw)
        state["initial_raw_grad_inf"] = float(grad_norm_raw)

        # Also store them in param for transparency/debugging. The actual loss
        # uses state, so the values remain shared across copied stage params.
        param["objective_scale"] = float(objective_scale)
        param["gradient_post_scale"] = float(gradient_post_scale)

        scaling_info = {
            "auto_scale": auto_scale,
            "objective_scale": float(objective_scale),
            "gradient_post_scale": float(gradient_post_scale),
            "initial_raw_objective": float(objective_raw),
            "initial_raw_grad_inf": float(grad_norm_raw),
            "initial_scaled_objective": float(objective_raw) * float(objective_scale),
            "initial_scaled_grad_inf": float(grad_norm_raw) * float(gradient_post_scale),
            "objective_scale_target": float(param.get("objective_scale_target", 1e7)),
            "gradient_scale_target": float(param.get("gradient_scale_target", 1e-2)),
        }

        os.makedirs(param["save_dir"], exist_ok=True)
        with open(os.path.join(param["save_dir"], "scaling_info.json"), "w") as f:
            import json
            json.dump(scaling_info, f, indent=2)

        print(
            "[scaling initialized] "
            f"objective_scale = {objective_scale:.8e}, "
            f"gradient_post_scale = {gradient_post_scale:.8e}, "
            f"raw objective = {objective_raw:.8e}, "
            f"raw |grad|_inf = {grad_norm_raw:.8e}",
            flush=True,
        )

        return float(objective_scale), float(gradient_post_scale)


    def fwi_loss(x, param, state):
        """
        Loss function for scipy.optimize.minimize.

        The forward/adjoint computation returns raw objective and raw gradients.
        This function then applies fixed scalar objective/gradient scales.

        With auto_scale=True, the scales are computed only at loss_call == 0:
            objective_scale      = objective_scale_target / |objective_raw|,
            gradient_post_scale = gradient_scale_target / |grad_x_raw|_inf.

        Those two scales are then frozen for all later line-search evaluations
        and for both L-BFGS-B stages.
        """
        import os
        import copy

        # ------------------------------------------------------------
        # 1. Convert optimization variable x -> full vp/vs model.
        # ------------------------------------------------------------
        vp, vs = x_to_vpvs(
            x,
            shape=state["shape"],
            param=param,
            vp_water=state["vp_water"],
            vs_water=state["vs_water"],
        )

        # ------------------------------------------------------------
        # 2. Save current trial model.
        # ------------------------------------------------------------
        trial_model_dir = os.path.join(param["work_dir"], "trial_model")
        os.makedirs(trial_model_dir, exist_ok=True)

        np.save(os.path.join(trial_model_dir, "vp.npy"), vp.astype(np.float32))
        np.save(os.path.join(trial_model_dir, "vs.npy"), vs.astype(np.float32))

        rho = state["rho"]
        if np.ndim(rho) == 2:
            np.save(os.path.join(trial_model_dir, "rho.npy"), rho.astype(np.float32))

        # ------------------------------------------------------------
        # 3. Current trial param points to trial_model.
        # ------------------------------------------------------------
        p = copy.deepcopy(param)
        p["model"]["model_path"] = trial_model_dir

        # Gradient computation requires saved forward displacement wavefield.
        p["solver"]["save_u"] = True

        # The stage controls whether inv_multi_shots smooths the cutoff gradient.
        p["smooth_gradient"] = bool(state.get("stage_smooth_gradient", True))

        # ------------------------------------------------------------
        # 4. Compute raw objective and lambda/mu gradients by multi-shot FWI.
        # ------------------------------------------------------------
        inv_out = inv_multi_shots(p)

        objective_raw = float(inv_out["objective"])
        grad_lam = np.asarray(inv_out["grad_lam"], dtype=np.float64)
        grad_mu = np.asarray(inv_out["grad_mu"], dtype=np.float64)

        # ------------------------------------------------------------
        # 5. Convert grad_lam / grad_mu -> raw grad_x.
        # ------------------------------------------------------------
        grad_x_raw = lame_grad_to_x_grad(
            grad_lam=grad_lam,
            grad_mu=grad_mu,
            vp=vp,
            vs=vs,
            rho=rho,
            param=param,
        )

        # ------------------------------------------------------------
        # 6. Initialize/reuse scaling and apply it.
        # ------------------------------------------------------------
        objective_scale, gradient_post_scale = initialize_or_get_scales(
            objective_raw=objective_raw,
            grad_x_raw=grad_x_raw,
            param=param,
            state=state,
        )

        objective = objective_scale * objective_raw
        grad_x = gradient_post_scale * grad_x_raw

        # ------------------------------------------------------------
        # 7. Record objective only.
        # ------------------------------------------------------------
        state["last_objective"] = float(objective)

        # Only record the initial objective on the first loss call. Accepted
        # iterations are recorded in callback().
        if state["loss_call"] == 0:
            state["objective_history"].append(float(objective))
            np.save(
                os.path.join(param["save_dir"], "objective_history.npy"),
                np.asarray(state["objective_history"], dtype=np.float64),
            )

        stage_id = state.get("stage_id", 1)
        stage_name = state.get("stage_name", "single")
        smooth_gradient = bool(state.get("stage_smooth_gradient", True))

        print(
            f"[loss {state['loss_call']:04d} | "
            f"stage {stage_id}:{stage_name} | smooth={smooth_gradient}] "
            f"objective = {objective:.8e} "
            f"(raw {objective_raw:.8e}), "
            f"|grad|_inf = {np.max(np.abs(grad_x)):.8e} "
            f"(raw {np.max(np.abs(grad_x_raw)):.8e})",
            flush=True,
        )

        state["loss_call"] += 1

        return float(objective), grad_x.astype(np.float64)

    def fwi_callback(xk, param, state):
        """
        Callback called by scipy.optimize.minimize after each accepted iteration.

        Saves only:
            vp.npy
            vs.npy
            objective_history.npy

        The accepted iteration number is global across all stages, so with
        stages 40 + 40 the saved folders are still iter_0001, ..., iter_0080.
        """
        import os
        import json

        iter_id = state["accepted_iter"]
        stage_id = int(state.get("stage_id", 1))
        stage_name = str(state.get("stage_name", "single"))
        smooth_gradient = bool(state.get("stage_smooth_gradient", True))

        vp, vs = x_to_vpvs(
            xk,
            shape=state["shape"],
            param=param,
            vp_water=state["vp_water"],
            vs_water=state["vs_water"],
        )

        iter_dir = os.path.join(param["save_dir"], f"iter_{iter_id:04d}")
        os.makedirs(iter_dir, exist_ok=True)

        np.save(os.path.join(iter_dir, "vp.npy"), vp.astype(np.float32))
        np.save(os.path.join(iter_dir, "vs.npy"), vs.astype(np.float32))

        meta = {
            "accepted_iter": iter_id,
            "stage_id": stage_id,
            "stage_name": stage_name,
            "smooth_gradient": smooth_gradient,
            "objective": None if state["last_objective"] is None else float(state["last_objective"]),
            "objective_scale": None if state.get("objective_scale") is None else float(state["objective_scale"]),
            "gradient_post_scale": None if state.get("gradient_post_scale") is None else float(state["gradient_post_scale"]),
        }
        with open(os.path.join(iter_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        if state["last_objective"] is not None:
            state["objective_history"].append(float(state["last_objective"]))
        else:
            print("[callback warning] last_objective is None, objective not recorded.", flush=True)

        # Save objective history globally.
        np.save(
            os.path.join(param["save_dir"], "objective_history.npy"),
            np.asarray(state["objective_history"], dtype=np.float64),
        )

        print(
            f"[callback] saved iter_{iter_id:04d} "
            f"from stage {stage_id}:{stage_name}, smooth={smooth_gradient}",
            flush=True,
        )

        state["accepted_iter"] += 1


    def fwi(param):
        """
        Run two-stage FWI using optimization variables:

            x = [vs_reduced, var_reduced]
            var = vp - eta * vs

        With the current param['fwi_stages'], this performs two independent
        L-BFGS-B runs:
            stage 1: 40 iterations with Gaussian-smoothed gradients;
            stage 2: 40 iterations without Gaussian smoothing.

        The L-BFGS-B memory is therefore reset between stages, but the model,
        objective/gradient scales, iteration counter, and objective history are
        carried over continuously.
        """
        import os
        import copy
        import json
        from scipy import optimize

        # ------------------------------------------------------------
        # 1. Prepare directories.
        # ------------------------------------------------------------
        os.makedirs(param["work_dir"], exist_ok=True)
        os.makedirs(param["save_dir"], exist_ok=True)

        # ------------------------------------------------------------
        # 2. Load initial model.
        # ------------------------------------------------------------
        model0 = build_model_from_param(param)

        vp0 = np.asarray(model0.vp_array, dtype=np.float64)
        vs0 = np.asarray(model0.vs_array, dtype=np.float64)
        rho0 = np.asarray(model0.rho_array, dtype=np.float64)

        shape = vp0.shape

        # Homogeneous fixed water layer values.
        vp_water = float(vp0[0, 0])
        vs_water = float(vs0[0, 0])

        print(f"[FWI] model shape      : {shape}", flush=True)
        print(f"[FWI] water_layer      : {param['water_layer']}", flush=True)
        print(f"[FWI] vp_water / vs_water : {vp_water:.6e} / {vs_water:.6e}", flush=True)

        # ------------------------------------------------------------
        # 3. Initial optimization vector.
        # ------------------------------------------------------------
        x_current = vpvs_to_x(vp0, vs0, param)

        # ------------------------------------------------------------
        # 4. Save initial model.
        # ------------------------------------------------------------
        init_dir = os.path.join(param["save_dir"], "iter_0000_initial")
        os.makedirs(init_dir, exist_ok=True)

        np.save(os.path.join(init_dir, "vp.npy"), vp0.astype(np.float32))
        np.save(os.path.join(init_dir, "vs.npy"), vs0.astype(np.float32))

        # ------------------------------------------------------------
        # 5. Lightweight state shared by all stages.
        # ------------------------------------------------------------
        state = {
            "shape": shape,
            "vp_water": vp_water,
            "vs_water": vs_water,
            "rho": rho0,

            # loss_call counts all objective evaluations, including line search.
            "loss_call": 0,

            # accepted_iter counts accepted optimizer steps globally.
            # 0 is reserved for iter_0000_initial.
            "accepted_iter": 1,

            "objective_history": [],
            "last_objective": None,

            # Scaling is initialized by the first fwi_loss call and then frozen.
            "scale_initialized": False,
            "objective_scale": None,
            "gradient_post_scale": None,

            # Updated before each stage.
            "stage_id": 0,
            "stage_name": None,
            "stage_smooth_gradient": True,
        }

        # ------------------------------------------------------------
        # 6. Build bounds and stage config.
        # ------------------------------------------------------------
        bounds = build_fwi_bounds(x_current, param)
        stages = get_fwi_stages(param)

        with open(os.path.join(param["save_dir"], "fwi_stages.json"), "w") as f:
            json.dump(stages, f, indent=2)

        print("[FWI] stages:", flush=True)
        for i, stage in enumerate(stages, start=1):
            print(
                f"    stage {i}: name={stage['name']}, "
                f"maxiter={stage['maxiter']}, "
                f"smooth_gradient={stage['smooth_gradient']}",
                flush=True,
            )

        # ------------------------------------------------------------
        # 7. Run stage-wise L-BFGS-B.
        # ------------------------------------------------------------
        stage_results = []
        stage_summaries = []
        total_stage_iter = 0

        for stage_id, stage in enumerate(stages, start=1):
            stage_name = stage["name"]
            stage_maxiter = int(stage["maxiter"])
            stage_smooth_gradient = bool(stage["smooth_gradient"])
            total_stage_iter += stage_maxiter

            state["stage_id"] = stage_id
            state["stage_name"] = stage_name
            state["stage_smooth_gradient"] = stage_smooth_gradient

            # Each stage gets its own param copy so the stage-specific smooth flag
            # is explicit and cannot leak backward unexpectedly.
            p_stage = copy.deepcopy(param)
            p_stage["smooth_gradient"] = stage_smooth_gradient

            print(
                f"[FWI] starting stage {stage_id}:{stage_name} | "
                f"maxiter={stage_maxiter}, smooth_gradient={stage_smooth_gradient}",
                flush=True,
            )

            result = optimize.minimize(
                fun=lambda x, pp=p_stage: fwi_loss(x, pp, state),
                x0=x_current,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                callback=lambda xk, pp=p_stage: fwi_callback(xk, pp, state),
                options={
                    "maxiter": stage_maxiter,
                    "maxls": int(param.get("maxls", 20)),
                    "maxcor": int(param.get("maxcor", 10)),
                    "ftol": float(param.get("ftol", 1e-8)),
                    "disp": bool(param.get("disp", True)),
                },
            )

            stage_results.append(result)
            x_current = np.asarray(result.x, dtype=np.float64)

            # Save the model at the end of this stage, even if scipy reports
            # success=False due only to reaching maxiter.
            vp_stage, vs_stage = x_to_vpvs(
                x_current,
                shape=shape,
                param=param,
                vp_water=vp_water,
                vs_water=vs_water,
            )
            stage_dir = os.path.join(param["save_dir"], f"stage_{stage_id:02d}_{stage_name}_final")
            os.makedirs(stage_dir, exist_ok=True)
            np.save(os.path.join(stage_dir, "vp.npy"), vp_stage.astype(np.float32))
            np.save(os.path.join(stage_dir, "vs.npy"), vs_stage.astype(np.float32))

            stage_info = {
                "stage_id": stage_id,
                "stage_name": stage_name,
                "maxiter": stage_maxiter,
                "smooth_gradient": stage_smooth_gradient,
                "success": bool(result.success),
                "message": str(result.message),
                "nit": int(getattr(result, "nit", -1)),
                "nfev": int(getattr(result, "nfev", -1)),
                "objective_scale": None if state.get("objective_scale") is None else float(state["objective_scale"]),
                "gradient_post_scale": None if state.get("gradient_post_scale") is None else float(state["gradient_post_scale"]),
            }
            with open(os.path.join(stage_dir, "stage_info.json"), "w") as f:
                json.dump(stage_info, f, indent=2)
            stage_summaries.append(stage_info)

            print(
                f"[FWI] finished stage {stage_id}:{stage_name} | "
                f"success={result.success}, nit={getattr(result, 'nit', None)}, "
                f"message={result.message}",
                flush=True,
            )

            if not np.all(np.isfinite(x_current)):
                raise FloatingPointError(
                    f"Non-finite model vector encountered after stage {stage_id}:{stage_name}."
                )

        # ------------------------------------------------------------
        # 8. Save final model.
        # ------------------------------------------------------------
        vp_final, vs_final = x_to_vpvs(
            x_current,
            shape=shape,
            param=param,
            vp_water=vp_water,
            vs_water=vs_water,
        )

        final_dir = os.path.join(param["save_dir"], "final")
        os.makedirs(final_dir, exist_ok=True)

        np.save(os.path.join(final_dir, "vp.npy"), vp_final.astype(np.float32))
        np.save(os.path.join(final_dir, "vs.npy"), vs_final.astype(np.float32))

        np.save(
            os.path.join(param["save_dir"], "objective_history.npy"),
            np.asarray(state["objective_history"], dtype=np.float64),
        )

        # Return the final scipy OptimizeResult for compatibility, while
        # attaching compact summaries. Avoid storing full OptimizeResult objects
        # inside final_result because that can create self-referential reprs.
        final_result = stage_results[-1]
        final_result["x"] = x_current
        final_result["stage_summaries"] = stage_summaries
        final_result["fwi_state_summary"] = {
            "loss_call": int(state["loss_call"]),
            "accepted_iter": int(state["accepted_iter"]),
            "objective_scale": None if state.get("objective_scale") is None else float(state["objective_scale"]),
            "gradient_post_scale": None if state.get("gradient_post_scale") is None else float(state["gradient_post_scale"]),
            "initial_raw_objective": None if state.get("initial_raw_objective") is None else float(state["initial_raw_objective"]),
            "initial_raw_grad_inf": None if state.get("initial_raw_grad_inf") is None else float(state["initial_raw_grad_inf"]),
            "final_objective": None if not state["objective_history"] else float(state["objective_history"][-1]),
        }
        final_result["total_requested_stage_iter"] = total_stage_iter

        print("[FWI] finished all stages", flush=True)
        print(f"[FWI] final-stage success : {final_result.success}", flush=True)
        print(f"[FWI] final-stage message : {final_result.message}", flush=True)
        print(f"[FWI] objective_scale     : {state['objective_scale']}", flush=True)
        print(f"[FWI] gradient_post_scale: {state['gradient_post_scale']}", flush=True)

        if state["objective_history"]:
            print(f"[FWI] final objective : {state['objective_history'][-1]:.8e}", flush=True)

        return final_result

    try:
        is_first_call = True
        import time
        ts = time.time()
        result = fwi(param)
        te = time.time()
        print("FWI result:")
        print(result)
        print(f"Inversion time: {te-ts}")

    finally:
        client.close()
        cluster.close()