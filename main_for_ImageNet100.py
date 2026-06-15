import json
import os
import subprocess
import copy
import sys
from concurrent.futures import ProcessPoolExecutor
import time
import logging
# -------------------------------- SETTINGS ------------------------------
VERSION = "v2"
MODEL_NAME = "cadc"
DATASET = "ImageNet100"
CONFIG_FILE = "exps/{}_{}.json".format(MODEL_NAME,DATASET)
SEEDS = [1993, 2017, 2020]  #  [1993, 2017, 2020] 
INCREMENTS= [(5, 5), (10, 10), (20, 20)]  # (5, 5), (10, 10), (20, 20)
GPUS = [0, 1]
MAX_CONCURRENT_PROCESSES = len(GPUS)

Description = "final"
# --------------------------RECOMMEND PARAMS FOR EACH TASKS---------------
"""
CIFAR100 
-------------10 phases----------

-------------5 phases----------

-------------20 phases----------

"""
# ------------------------------------------------------------------------
param_grid_diff_tasks = {
    "5_5":[
        {"w_kd":10, "cosine": True, "drift_scale": 1.0,  "batch_size": 64},
    ],
    "10_10": [
        {"w_kd":10, "cosine": True, "drift_scale": 1.0,  "batch_size": 64},
    ],
    "20_20":[
        {"w_kd":10, "cosine": True, "drift_scale": 1.0,  "batch_size": 64},
    ]
}
# ------------------------ LOGGING CONFIGURATION ------------------------
log_dir = f"logs/"
log_file = os.path.join(log_dir, f"{MODEL_NAME}_tune_{DATASET}.log")

if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)  # exist_ok
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(filename)s] => %(message)s",
    handlers=[
        logging.FileHandler(log_file),  
        logging.StreamHandler(sys.stdout),
    ],
)


def run_experiment(params, gpu_id, init_cls, increment, seed, process_id):
    # load config
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    # update config
    config_updated = copy.deepcopy(config)
    config_updated.update(params)

    config_updated['resume'] = False
    config_updated["device"] = [str(gpu_id)]
    config_updated["seed"] = [seed]
    config_updated["process_id"] = process_id

    config_updated["init_cls"] = init_cls
    config_updated["increment"] = increment
    config_updated["note"] = Description + str(params)
    config_updated["version"] = VERSION
    cmd = ["python", "main_tune.py"]
    env = os.environ.copy()
    logging.info(f"Running experiment on seed {seed}, init_cls {init_cls}, increm {increment}, on GPU {gpu_id}, params:{params}")
    env["CONFIG_JSON"] = json.dumps(config_updated)
    subprocess.run(cmd, env=env)


with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_PROCESSES) as executor:
    futures = []
    idx = 0
    task_count = 0  
    process_ids = list(range(MAX_CONCURRENT_PROCESSES))

    for init_cls, increment in INCREMENTS:
        for _, params in enumerate(param_grid_diff_tasks["{}_{}".format(init_cls, increment)]):
            for seed in SEEDS:
                gpu_id = GPUS[idx % len(GPUS)]
                idx += 1
                process_id = process_ids[task_count % MAX_CONCURRENT_PROCESSES]
                task_count += 1
                futures.append(executor.submit(run_experiment, params, gpu_id, init_cls, increment, seed, process_id))
                time.sleep(3)
    # wait for all futures to complete
    for future in futures:
        future.result()
