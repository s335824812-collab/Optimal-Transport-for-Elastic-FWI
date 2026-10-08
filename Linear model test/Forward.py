import numpy as np
import logging, os, sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from examples.seismic import Model, plot_velocity, plot_image

from dask.distributed import Client, LocalCluster, wait, progress

from devito import configuration
configuration["log-level"] = "WARNING"

import matplotlib.pyplot as plt

from devito import *

import time

from Elastic_pml import *

if __name__ == '__main__':
    param = {
        "model":{
            'model_path': './Model',
            'spacing': (10.,10.),
            'origin': (0.,0.)
        },
        
        "pml":{
            'npml': 20
        },
        
        "geometry":{
            'nshots':40,
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
            'save_u': False,
            'u_save_interval': 10,
            'configure_devito': False,
        },
        
        "save_path": './Data',
        "dask_nwrks": 10
    }
    model = build_model_from_param(param)
    extent = (model.vp.shape[0]-1)*param['model']['spacing'][0]
    src_x = np.linspace(20., extent-20., param['geometry']['nshots'])
    rec_x = np.linspace(20., extent-20., param['geometry']['nrec'])
    param["geometry"]['src_positions'] = np.column_stack([src_x, np.full_like(src_x, param['geometry']['src_depth'])])
    param["geometry"]['rec_positions'] = np.column_stack([rec_x, np.full_like(rec_x, param['geometry']['rec_depth'])])
    
    print("Initializing local cluster...")
    cluster = LocalCluster(
        n_workers=param['dask_nwrks'],
        threads_per_worker=1,
        processes=True,
        silence_logs=logging.INFO,
        )
    client = Client(cluster)
    time.sleep(10)
    
    def forward_single_shot(param):
        try:
            model = build_model_from_param(param)
            geometry_s = build_geometry_from_param(model, param)
            solver = build_forward_solver_from_param(model, geometry_s, param)
            out = solver.forward()
            np.save(param['save_path'], np.array([out['ux']]+ [out['uz']]))
            return
        except Exception as e:
            print(f"Error computing gradient for shot {param['shot_id']}: {e}")
            return forward_single_shot(param)
        

    def forward_multi_shots(param):
        import copy

        work = [copy.deepcopy(param) for _ in range(param["geometry"]["nshots"])]
        futures = []
        for s in range(param['geometry']['nshots']):
            work[s]['shot_id'] = s
            work[s]['geometry']['src_positions'] = param['geometry']['src_positions'][s,:]
            fname = f"Data_for_{s}_shot_{int(param['geometry']['f0']*1e3)}hz.npy"
            work[s]['save_path'] = os.path.join(param['save_path'], fname)
        futures = client.map(forward_single_shot, work)
        progress(futures)
        wait(futures)
        return
    
    ts = time.time()
    data = forward_multi_shots(param)
    te = time.time()
    et = te - ts
    print("done")
    print(et)
    client.close()
    cluster.close()