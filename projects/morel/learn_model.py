"""
Script to learn MDP model from data for offline policy optimization
"""

from os import environ

environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
environ["MKL_THREADING_LAYER"] = "GNU"

DEFAULT_FOLDER = "dynamics"
import numpy as np
import copy
import torch
import torch.nn as nn
import pickle
import mjrl.envs
import time
import argparse
import os
import json
import mjrl.samplers.core as sampler
import mjrl.utils.tensor_utils as tensor_utils
from tqdm import tqdm
from tabulate import tabulate
from datetime import datetime
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.utils.gym_env import GymEnv
from mjrl.utils.logger import DataLog
from mjrl.utils.make_train_plots import make_train_plots
from mjrl.algos.mbrl.nn_dynamics import WorldModel
from mjrl.algos.mbrl.model_based_npg import ModelBasedNPG
from mjrl.algos.mbrl.sampling import sample_paths, evaluate_policy

# ===============================================================================
# Get command line arguments
# ===============================================================================

parser = argparse.ArgumentParser(description="Model accelerated policy optimization.")
parser.add_argument(
    "--output",
    "-o",
    type=str,
    required=True,
    help="location to store the model pickle file",
)
parser.add_argument(
    "--config",
    "-c",
    type=str,
    required=True,
    help="path to config file with exp params",
)
parser.add_argument(
    "--include", "-i", type=str, required=False, help="package to import"
)
args = parser.parse_args()
with open(args.config, "r") as f:
    job_data = eval(f.read())
if args.include:
    exec("import " + args.include)

assert "data_folder" in job_data.keys()
ENV_NAME = job_data["env_name"]
SEED = job_data["seed"]
del job_data["seed"]
if "act_repeat" not in job_data.keys():
    job_data["act_repeat"] = 1

# Output folder and model
output_folder = os.path.join(
    DEFAULT_FOLDER, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_model = os.path.join(output_folder, args.output)

# ===============================================================================
# Construct environment and model
# ===============================================================================

from ac_pulse import ACPulse

env = ACPulse({"obs_shape": 80})

if ENV_NAME.split("_")[0] == "dmc":
    # import only if necessary (not part of package requirements)
    import dmc2gym

    backend, domain, task = ENV_NAME.split("_")
    e = dmc2gym.make(domain_name=domain, task_name=task, seed=SEED)
    e = GymEnv(e, act_repeat=job_data["act_repeat"])
else:
    e = GymEnv(env, act_repeat=job_data["act_repeat"])
    e.set_seed(SEED)

models = [
    WorldModel(
        state_dim=e.observation_dim, act_dim=e.action_dim, seed=SEED + i, **job_data
    )
    for i in range(job_data["num_models"])
]

# ===============================================================================
# Model training loop
# ===============================================================================

# '''
#     For the given path, get the List of all files in the directory tree
# '''
def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def refresh_dataset(reader):
    batch = reader.next().data
    print("Collecting batches of data from file . . .")

    pbar = tqdm(
        unit=" transitions", desc="Transitions collected"
    )  # create progress bar

    # batch["terminals"] = batch.pop("dones")
    # batch["observations"] = batch.pop("obs")
    # batch["next_observations"] = batch.pop("new_obs")

    # Reshaping arrays that come with one dimension so they have (size, 1)
    # New shape should be (size, action_shape)
    ac = np.array((len(batch["actions"]), 2))
    for i in range(len(batch["actions"])):
        ac[i][batch["actions"][i]] = 1
    batch["actions"] = ac

    batch["rewards"] = batch["rewards"].reshape((batch["rewards"].size, 1))
    batch["dones"] = batch["dones"].reshape((batch["dones"].size, 1))

    s = batch["obs"]
    a = batch["actions"]
    sp = batch["new_obs"]
    r = batch["rewards"]
    pbar.update(r.size)  # update bar

    while r.size < 1e6:
        batch = reader.next().data

        # New shape should be (size, action_shape)]
        ac = np.array((len(batch["actions"]), 2))
        for i in range(len(batch["actions"])):
            ac[i][batch["actions"][i]] = 1
        batch["actions"] = ac

        batch["rewards"] = batch["rewards"].reshape((batch["rewards"].size, 1))
        batch["dones"] = batch["dones"].reshape((batch["dones"].size, 1))

        # Concatenating the iteractions to create one huge dataset in numpy arrays
        s = np.concatenate((s, batch["obs"]))
        a = np.concatenate((a, batch["actions"]))
        sp = np.concatenate((sp, batch["new_obs"]))
        r = np.concatenate((r, batch["rewards"]))

        pbar.update(batch["rewards"].size)  # update bar
    pbar.close()
    timer.sleep(0.2)

    return s, a, sp, r


from ray.rllib.offline import JsonReader

dataList = getListOfFiles(job_data["data_folder"])
reader = JsonReader(dataList)

# The dataset might be too big to hold on memory, thus we will collect a part of it, train, than collect another, and so on
# Define how many times the dataset will be refreshed and how many epochs will be passed to the fit functions
training_rounds = job_data["refresh_dataset_times"]
if job_data["fit_epochs"] % job_data["refresh_dataset_times"] > 0:
    print(
        "WARNING: Total epochs (FIT_EPOCHS) is not divisible by REFRESH_DATASET_TIMES, FIT_EPOCHS will be reduced to {}".format(
            job_data["fit_epochs"]
            - (job_data["fit_epochs"] % job_data["refresh_dataset_times"])
        )
    )
    total_epochs = job_data["fit_epochs"] - (
        job_data["fit_epochs"] % job_data["refresh_dataset_times"]
    )
else:
    total_epochs = job_data["fit_epochs"]

job_data["fit_epochs"] = job_data["fit_epochs"] // job_data["refresh_dataset_times"]
# -------------

# Tensordboard writer
from tensorboardX import SummaryWriter

writer = SummaryWriter(output_folder)

for training_round in range(training_rounds):
    time.sleep(0.2)
    print("\n")
    print("Staring round {} of {}".format(training_round + 1, training_rounds))
    # print("{} / {} epochs")

    for i, model in enumerate(models):
        s, a, sp, r = refresh_dataset(reader)
        print("\n")
        print("\nDynamics model {}\n".format(i))

        dynamics_loss = model.fit_dynamics(s, a, sp, **job_data)
        loss_general = model.compute_loss(s, a, sp)  # generalization error

        if job_data["learn_reward"]:
            print("\nReward model {}\n".format(i))
            reward_loss = model.fit_reward(s, a, r.reshape(-1, 1), **job_data)
            # Logging
            for j, l in enumerate(reward_loss):
                writer.add_scalar(
                    "model_{}/reward_loss".format(i),
                    l,
                    j + (job_data["fit_epochs"] * (training_round)),
                )

        # Logging
        for j, l in enumerate(dynamics_loss):
            writer.add_scalar(
                "model_{}/dynamics_loss".format(i),
                l,
                j + (job_data["fit_epochs"] * (training_round)),
            )
        writer.add_scalar(
            "model_{}/generalized_loss".format(i),
            loss_general,
            training_round,
        )

        del s, a, sp, r  # just to release the memory in case they are too big

pickle.dump(models, open(output_model, "wb"))
