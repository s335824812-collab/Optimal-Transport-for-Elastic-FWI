import os
import sys
import time
import copy
import logging
from pathlib import Path

# =========================
# Thread control: put before heavy imports
# =========================
os.environ.pop("CC", None)
os.environ.pop("CXX", None)
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

# =========================
# Add project root
# get_forward_data.py 位于 Test_4/Data_pml/Init_data
# Elastic_pml.py 位于 Test_4
# 所以 parents[2] 到 Test_4
# =========================
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from dask.distributed import Client, LocalCluster, wait, progress

from devito import configuration
configuration["log-level"] = "WARNING"

import Elastic_pml as epml
# import Elastic_abc as eabc   


def make_param():
    param = {
        "model": {
            "model_path": "../../Model_1200",
            "spacing": (10., 10.),
            "origin": (0., 0.),
        },

        "pml": {
            "npml": 30,
            'vpmax': 5.0,
        },

        "geometry": {
            "hor_extent_padding": 1200,
            "nshots": 97,
            "dshots": 100,
            "nrec": 321,
            "drec": 30,
            "src_depth": 30,
            "rec_depth": 70,
            "t0": 0.0,
            "tn": 6000.0,
            "dt": 0.75,
            "f0": 0.010,
            "receiver_components": ("ux", "uz"),
        },

        "solver": {
            "space_order": 4,
            "time_order": 1,
            "cfl": 0.4,
            "save_u": False,
            "u_save_interval": 10,
            "configure_devito": False,
        },

        "save_path": "./",
        "dask_nwrks": 10,
    }

    src_x = (
        param["geometry"]["hor_extent_padding"]
        + param["geometry"]["dshots"] * np.arange(param["geometry"]["nshots"])
    )

    rec_x = (
        param["geometry"]["hor_extent_padding"]
        + param["geometry"]["drec"] * np.arange(param["geometry"]["nrec"])
    )

    param["geometry"]["src_positions"] = np.column_stack([
        src_x,
        np.full_like(src_x, param["geometry"]["src_depth"])
    ])

    param["geometry"]["rec_positions"] = np.column_stack([
        rec_x,
        np.full_like(rec_x, param["geometry"]["rec_depth"])
    ])

    return param


def _to_numpy_data(x):
    """
    尽量兼容 Devito Receiver / numpy array。
    """
    if hasattr(x, "data"):
        return np.asarray(x.data)
    return np.asarray(x)


def forward_single_shot(param):
    shot_id = param.get("shot_id", -1)

    try:
        model = epml.build_model_from_param(param)
        geometry_s = epml.build_geometry_from_param(model, param)
        solver = epml.build_forward_solver_from_param(model, geometry_s, param)

        out = solver.forward()

        ux = _to_numpy_data(out["ux"])
        uz = _to_numpy_data(out["uz"])

        data = np.stack([ux, uz], axis=0)
        np.save(param["save_path"], data)

        return {
            "shot_id": shot_id,
            "status": "ok",
            "save_path": param["save_path"],
        }

    except Exception as e:
        # 不建议递归 return forward_single_shot(param)
        # 否则某个 shot 一直错会无限递归
        return {
            "shot_id": shot_id,
            "status": "failed",
            "error": repr(e),
        }


def forward_multi_shots(param, client):
    work = []

    for s in range(param["geometry"]["nshots"]):
        p = copy.deepcopy(param)

        p["shot_id"] = s

        # 注意这里用 s:s+1 保持二维形状 (1, 2)
        # 不要用 [s, :] 变成一维 (2,)
        p["geometry"]["src_positions"] = param["geometry"]["src_positions"][s:s+1, :]

        fname = f"Data_for_{s}_shot_{int(param['geometry']['f0'] * 1e3)}hz.npy"
        p["save_path"] = os.path.join(param["save_path"], fname)

        work.append(p)

    futures = client.map(forward_single_shot, work, resources={'SHOT_VOL': 1})
    progress(futures)
    wait(futures)

    results = client.gather(futures)

    failed = [r for r in results if r["status"] != "ok"]
    if failed:
        print("\nSome shots failed:")
        for r in failed:
            print(f"shot {r['shot_id']}: {r['error']}")
    else:
        print("\nAll shots finished successfully.")

    return results


def main():
    param = make_param()

    print("Initializing local cluster...")

    from os.path import join
    from dask_jobqueue import SLURMCluster
    # np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

    dir_pre = '/fs2/home/ymsc_cpu/homes/heyuxuan/Elastic_fwi/Marmousi_pml_revised_old_model'
    ## setup cluster. The following configuration can also be put into ~/.config/dask/jobqueue.yaml
    cluster = SLURMCluster(cores=56,      # Total number of cores per job
                        processes=2,      # Cut the job up into this many processes
                        memory="256GB",   # Total amount of memory per job
                        walltime="24:00:00",
                        queue="cp6",
                        interface="gn0",  # for fast network interface ib0
                        # local_directory="/fs2/home/ymsc_cpu/homes/Lingyun/running/homogenization_sediment_measure/tmp" ,
                        local_directory="/tmp" ,
                        log_directory=join(dir_pre,"logs"),
                        #job_extra=['--exclude=c0320,c0323,c0327,c0311'],   # List of other Slurm options
                        extra=['--resources SHOT_VOL=1']
                        )
    cluster.scale(param['dask_nwrks'])
    client = Client(cluster)
    time.sleep(20)

    try:
        ts = time.time()

        results = forward_multi_shots(param, client)

        te = time.time()
        print("done")
        print("elapsed time:", te - ts)

    finally:
        try:
            client.close()
        except Exception as e:
            print("client.close warning:", repr(e))

        try:
            cluster.close()
        except Exception as e:
            print("cluster.close warning:", repr(e))


if __name__ == "__main__":
    main()