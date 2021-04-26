"""
Job script to learn policy using MOReL
"""

from os import environ

environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
environ["MKL_THREADING_LAYER"] = "GNU"

DEFAULT_FOLDER = "models"
import numpy as np
import copy
import torch
import torch.nn as nn
import pandas as pd
import pickle
import mjrl.envs
import time as timer
import argparse
import os
import io
import json
import mjrl.samplers.core as sampler
import mjrl.utils.tensor_utils as tensor_utils
from tqdm import tqdm
from tabulate import tabulate
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.utils.gym_env import GymEnv
from mjrl.utils.logger import DataLog
from mjrl.utils.make_train_plots import make_train_plots
from mjrl.algos.mbrl.nn_dynamics import WorldModel
from mjrl.algos.mbrl.model_based_npg import ModelBasedNPG
from mjrl.algos.mbrl.sampling import sample_paths, evaluate_policy

from is_estimator import ImportanceSamplingEstimator
from custom_estimators import CustomImportanceSamplingEstimator
from custom_estimators import get_registry_reward, get_deal_reward, get_ltv_reward

import utils
from sklearn.metrics import confusion_matrix
from datetime import datetime

# import tensorflow as tf
from PIL import Image

# ===============================================================================
# Get command line arguments
# ===============================================================================

parser = argparse.ArgumentParser(description="Model accelerated policy optimization.")
parser.add_argument(
    "--output", "-o", type=str, required=True, help="location to store results"
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
OUT_DIR = os.path.join(DEFAULT_FOLDER, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
if not os.path.exists(OUT_DIR + "/iterations"):
    os.makedirs(OUT_DIR + "/iterations")
if not os.path.exists(OUT_DIR + "/logs"):
    os.makedirs(OUT_DIR + "/logs")
with open(args.config, "r") as f:
    job_data = eval(f.read())
if args.include:
    exec("import " + args.include)

# Unpack args and make files for easy access
logger = DataLog()
ENV_NAME = job_data["env_name"]
EXP_FILE = OUT_DIR + "/job_data.json"
SEED = job_data["seed"]

# base cases
if "eval_rollouts" not in job_data.keys():
    job_data["eval_rollouts"] = 0
if "save_freq" not in job_data.keys():
    job_data["save_freq"] = 10
if "device" not in job_data.keys():
    job_data["device"] = "cpu"
if "hvp_frac" not in job_data.keys():
    job_data["hvp_frac"] = 1.0
if "start_state" not in job_data.keys():
    job_data["start_state"] = "init"
if "learn_reward" not in job_data.keys():
    job_data["learn_reward"] = True
if "num_cpu" not in job_data.keys():
    job_data["num_cpu"] = 1
if "npg_hp" not in job_data.keys():
    job_data["npg_hp"] = dict()
if "act_repeat" not in job_data.keys():
    job_data["act_repeat"] = 1
if "model_file" not in job_data.keys():
    job_data["model_file"] = None

assert job_data["start_state"] in ["init", "buffer", "any"]
# assert "data_file" in job_data.keys()
with open(EXP_FILE, "w") as f:
    json.dump(job_data, f, indent=4)
del job_data["seed"]
job_data["base_seed"] = SEED


# ===============================================================================
# Helper functions
# ===============================================================================
# def buffer_size(paths_list):
#     return np.sum([p["observations"].shape[0] - 1 for p in paths_list])


# ===============================================================================
# Setup functions and environment
# ===============================================================================

np.random.seed(SEED)
torch.random.manual_seed(SEED)

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

# check for reward and termination functions
if "reward_file" in job_data.keys():
    import sys

    splits = job_data["reward_file"].split("/")
    dirpath = "" if splits[0] == "" else os.path.dirname(os.path.abspath(__file__))
    for x in splits[:-1]:
        dirpath = dirpath + "/" + x
    filename = splits[-1].split(".")[0]
    sys.path.append(dirpath)
    exec("from " + filename + " import *")
if "reward_function" not in globals():
    # reward_function = getattr(e.env.env, "compute_path_rewards", None)
    reward_function = None
    job_data["learn_reward"] = False if reward_function is not None else True
if "termination_function" not in globals():
    termination_function = None
if "obs_mask" in globals():
    e.obs_mask = obs_mask

# ===============================================================================
# Setup policy, model, and agent
# ===============================================================================

if job_data["model_file"] is not None:
    model_trained = True
    models = pickle.load(open(job_data["model_file"], "rb"))
else:
    model_trained = False
    models = [
        WorldModel(
            state_dim=e.observation_dim, act_dim=e.action_dim, seed=SEED + i, **job_data
        )
        for i in range(job_data["num_models"])
    ]

# Construct policy and set exploration level correctly for NPG
if "init_policy" in job_data.keys():
    policy = pickle.load(open(job_data["init_policy"], "rb"))
    policy.set_param_values(policy.get_param_values())
    init_log_std = job_data["init_log_std"]
    min_log_std = job_data["min_log_std"]
    if init_log_std:
        params = policy.get_param_values()
        params[: policy.action_dim] = tensor_utils.tensorize(init_log_std)
        policy.set_param_values(params)
    if min_log_std:
        policy.min_log_std[:] = tensor_utils.tensorize(min_log_std)
        policy.set_param_values(policy.get_param_values())
else:
    policy = MLP(
        e.spec,
        seed=SEED,
        hidden_sizes=job_data["policy_size"],
        init_log_std=job_data["init_log_std"],
        min_log_std=job_data["min_log_std"],
    )

# policy.set_transformations(out_shift=1.0, out_scale=0.5)

baseline = MLPBaseline(
    e.spec,
    reg_coef=1e-3,
    batch_size=256,
    epochs=1,
    learn_rate=1e-3,
    device=job_data["device"],
)
agent = ModelBasedNPG(
    learned_model=models,
    env=e,
    policy=policy,
    baseline=baseline,
    seed=SEED,
    normalized_step_size=job_data["step_size"],
    save_logs=True,
    reward_function=reward_function,
    termination_function=termination_function,
    **job_data["npg_hp"]
)

# ===============================================================================
# Model training loop
# ===============================================================================


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
    ac = np.zeros((len(batch["actions"]), 2))
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
        ac = np.zeros((len(batch["actions"]), 2))
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

best_perf = -1e8
ts = timer.time()
init_states_buffer = []
# this buffer is not used in the ac environment, thus the empty initialization

s, a, sp, r = refresh_dataset(reader)

# rollout_score = np.mean([np.sum(p["rewards"]) for p in paths])
num_samples = len(r)
logger.log_kv("fit_epochs", job_data["fit_epochs"])
# logger.log_kv("rollout_score", rollout_score)
logger.log_kv("iter_samples", num_samples)
logger.log_kv("num_samples", num_samples)
try:
    rollout_metric = e.env.env.evaluate_success(paths)
    logger.log_kv("rollout_metric", rollout_metric)
except:
    pass
if not model_trained:
    for i, model in enumerate(models):
        dynamics_loss = model.fit_dynamics(s, a, sp, **job_data)
        logger.log_kv("dyn_loss_" + str(i), dynamics_loss[-1])
        loss_general = model.compute_loss(s, a, sp)  # generalization error
        logger.log_kv("dyn_loss_gen_" + str(i), loss_general)
        if job_data["learn_reward"]:
            reward_loss = model.fit_reward(s, a, r.reshape(-1, 1), **job_data)
            logger.log_kv("rew_loss_" + str(i), reward_loss[-1])
else:
    for i, model in enumerate(models):
        loss_general = model.compute_loss(s, a, sp)
        logger.log_kv("dyn_loss_gen_" + str(i), loss_general)
tf = timer.time()
logger.log_kv("model_learning_time", tf - ts)
print("Model learning statistics")
print_data = sorted(
    filter(lambda v: np.asarray(v[1]).size == 1, logger.get_current_log().items())
)
print(tabulate(print_data))
pickle.dump(models, open(OUT_DIR + "/models.pickle", "wb"))
logger.log_kv(
    "act_repeat", job_data["act_repeat"]
)  # log action repeat for completeness

# ===============================================================================
# Pessimistic MDP parameters
# ===============================================================================

delta = np.zeros(s.shape[0])
for idx_1, model_1 in enumerate(models):
    pred_1 = model_1.predict(s, a)
    for idx_2, model_2 in enumerate(models):
        if idx_2 > idx_1:
            pred_2 = model_2.predict(s, a)
            disagreement = np.linalg.norm((pred_1 - pred_2), axis=-1)
            delta = np.maximum(delta, disagreement)

if "pessimism_coef" in job_data.keys():
    if job_data["pessimism_coef"] is None or job_data["pessimism_coef"] == 0.0:
        truncate_lim = None
        print("No pessimism used. Running naive MBRL.")
    else:
        truncate_lim = (1.0 / job_data["pessimism_coef"]) * np.max(delta)
        print(
            "Maximum error before truncation (i.e. unknown region threshold) = %f"
            % truncate_lim
        )
    job_data["truncate_lim"] = truncate_lim
    job_data["truncate_reward"] = (
        job_data["truncate_reward"] if "truncate_reward" in job_data.keys() else 0.0
    )
else:
    job_data["truncate_lim"] = None
    job_data["truncate_reward"] = 0.0

with open(EXP_FILE, "w") as f:
    job_data["seed"] = SEED
    json.dump(job_data, f, indent=4)
    del job_data["seed"]

# ===============================================================================
# Behavior Cloning Initialization
# ===============================================================================
if "bc_init" in job_data.keys():
    if job_data["bc_init"]:
        from mjrl.algos.behavior_cloning import BC

        policy.to(job_data["device"])
        bc_agent = BC(s, a, policy, epochs=5, batch_size=256, loss_type="MSE")
        bc_agent.train()

# ===============================================================================
# Policy Optimization Loop
# ===============================================================================


# --------------------------- Setup the evaluation process------------------------
is_estimator = ImportanceSamplingEstimator()
custom_estimator = CustomImportanceSamplingEstimator()

test_dataset_path = job_data["test_data"]
test_dataset = [
    os.path.join(test_dataset_path, f)
    for f in os.listdir(test_dataset_path)
    if os.path.isfile(os.path.join(test_dataset_path, f))
]
test_dataset = sorted(test_dataset)

rewards = []
for n_eps in range(len(test_dataset)):
    eval_reader = JsonReader(test_dataset[n_eps])

    with open(test_dataset[n_eps], "r") as f:
        sb = f.readlines()

    for _ in range(len(sb)):
        n = eval_reader.next()
        batch = eval_reader.next()
        for episode in batch.split_by_episode():
            for r in episode["rewards"]:
                rewards.append(r)

rewards_shift = (
    (round(min(rewards), 10) * -1) + 1e-6 if round(min(rewards), 10) <= 0 else 0
)
# ----------------------------------------------------------------------------------


from tensorboardX import SummaryWriter

writer = SummaryWriter(OUT_DIR)

for outer_iter in range(job_data["num_iter"]):
    ts = timer.time()
    agent.to(job_data["device"])
    if job_data["start_state"] == "init":
        print("sampling from initial state distribution")
        raise Exception(
            "This code cannot handle 'init' for 'start_state' yet as it requires knowing the number os episodes in the dataset"
        )
        # buffer_rand_idx = np.random.choice(len(init_states_buffer), size=job_data['update_paths'], replace=True).tolist()
        # init_states = [init_states_buffer[idx] for idx in buffer_rand_idx]
    elif job_data["start_state"] == "any":
        print("sampling starting states from the entire dataset")
        num_states_2 = job_data["update_paths"]
        buffer_rand_idx = np.random.choice(s.shape[0], size=num_states_2, replace=False)
        init_states_2 = list(s[buffer_rand_idx])
        init_states = init_states_2
    else:
        # Mix data between initial states and randomly sampled data from buffer
        print("sampling from mix of initial states and data buffer")
        raise Exception(
            "This code cannot handle values for 'start_state' other than 'any' "
        )
        if "buffer_frac" in job_data.keys():
            num_states_1 = (
                int(job_data["update_paths"] * (1 - job_data["buffer_frac"])) + 1
            )
            num_states_2 = int(job_data["update_paths"] * job_data["buffer_frac"]) + 1
        else:
            num_states_1, num_states_2 = (
                job_data["update_paths"] // 2,
                job_data["update_paths"] // 2,
            )
        buffer_rand_idx = np.random.choice(
            len(init_states_buffer), size=num_states_1, replace=True
        ).tolist()
        init_states_1 = [init_states_buffer[idx] for idx in buffer_rand_idx]
        buffer_rand_idx = np.random.choice(s.shape[0], size=num_states_2, replace=True)
        init_states_2 = list(s[buffer_rand_idx])
        init_states = init_states_1 + init_states_2

    train_stats = agent.train_step(
        N=len(init_states), init_states=init_states, **job_data
    )
    logger.log_kv("train_score", train_stats[0])
    agent.policy.to("cpu")

    # --------------------------------------------------------------------------------
    # evaluate true policy performance
    # --------------------------------------------------------------------------------
    print("Evaluating policy")
    actions = []
    true_actions = []

    estimation = {
        # "dm/score": [],
        # "dm/pred_reward_mean": [],
        # "dm/pred_reward_total": [],
        "is/V_prev": [],
        "is/V_step_IS": [],
        "is/V_gain_est": [],
    }
    custom_register = {
        "register/is/V_prev": [],
        "register/is/V_step_IS": [],
        "register/is/V_gain_est": [],
    }
    custom_deal = {
        "deal/is/V_prev": [],
        "deal/is/V_step_IS": [],
        "deal/is/V_gain_est": [],
    }
    custom_ltv = {
        "ltv/is/V_prev": [],
        "ltv/is/V_step_IS": [],
        "ltv/is/V_gain_est": [],
    }
    for n_eps in tqdm(range(len(test_dataset))):
        eval_reader = JsonReader(test_dataset[n_eps])
        batch = eval_reader.next()

        for episode in batch.split_by_episode():
            true_actions.extend(episode["actions"])

            # action, selected_action_prob, all_actions_prob = [], [], []
            # for i in range(len(episode["eps_id"])):
            #     _action = agent.get_action(episode["obs"][i])
            #     action.append(_action[1]["mean"])
            #     _action_prob = np.exp(
            #         agent.policy.log_likelihood(
            #             torch.from_numpy(np.asarray([episode["obs"][i]])),
            #             torch.from_numpy(np.asarray([episode["actions"][i]])),
            #         ),
            #     )

            #     # selected_action_prob.append(_action_prob)
            #     all_actions_prob.append(_action_prob)

            action = agent.policy.forward(episode["obs"]).argmax(axis=1)
            # print(action)
            # print(true_actions)
            # print(episode["obs"].shape)
            # print(episode["actions"].shape)
            # print(episode["actions"].reshape((-1, 1)).shape)
            # print("------")
            actions.extend(action)

            ac = np.zeros((len(episode["actions"]), 2))
            for i in range(len(episode["actions"])):
                ac[i][episode["actions"][i]] = 1
            # episode["actions"] = ac

            # print(ac)
            log_p = agent.policy.log_likelihood(
                torch.from_numpy(np.asarray(episode["obs"])), torch.from_numpy(ac)
            )
            # print(log_p)
            all_actions_prob = np.exp(
                log_p,
            )

            is_estimation = is_estimator.estimate(
                episode, all_actions_prob, rewards_shift
            )

            # Custom business metrics --------------------
            register_reward = get_registry_reward(episode)
            custom_register_estimate = custom_estimator.estimate(
                episode, all_actions_prob, register_reward
            )

            deal_reward = get_deal_reward(episode)
            custom_deal_estimate = custom_estimator.estimate(
                episode, all_actions_prob, deal_reward
            )

            ltv_reward = get_ltv_reward(episode, reward_shift=rewards_shift)
            custom_ltv_estimate = custom_estimator.estimate(
                episode, all_actions_prob, ltv_reward
            )

            # Custom Estimations -----------------------
            if custom_register_estimate:
                custom_register["register/is/V_prev"].append(
                    custom_register_estimate["V_prev"]
                )
                custom_register["register/is/V_step_IS"].append(
                    custom_register_estimate["V_step_IS"]
                )
                custom_register["register/is/V_gain_est"].append(
                    custom_register_estimate["V_gain_est"]
                )

            if custom_deal_estimate:
                custom_deal["deal/is/V_prev"].append(custom_deal_estimate["V_prev"])
                custom_deal["deal/is/V_step_IS"].append(
                    custom_deal_estimate["V_step_IS"]
                )
                custom_deal["deal/is/V_gain_est"].append(
                    custom_deal_estimate["V_gain_est"]
                )

            if custom_ltv_estimate:
                custom_ltv["ltv/is/V_prev"].append(custom_ltv_estimate["V_prev"])
                custom_ltv["ltv/is/V_step_IS"].append(custom_ltv_estimate["V_step_IS"])
                custom_ltv["ltv/is/V_gain_est"].append(
                    custom_ltv_estimate["V_gain_est"]
                )

            # Direct Estimator -----------------------
            # actions.extend(action)
            # action = np.array([action])
            # action_prob = np.array([selected_action_prob])

            # obs = torch.Tensor(
            #     np.concatenate(
            #         (episode["obs"], np.reshape(action, (action[0].shape[0], 1))),
            #         axis=1,
            #     )
            # )  # concatenate actions and observations for input obs are usually [[obs1],[obs2],[obs3]] and
            # # actions are usually [1,0,1,0] so the goal is to make actions like this: [[1],[0],[1]]
            # scores_raw = predictor.predict(obs).detach().numpy()
            # scores = {}
            # scores["score"] = (scores_raw * action_prob).mean()
            # scores["pred_reward_mean"] = scores_raw.mean()
            # scores["pred_reward_total"] = scores_raw.sum()

            # DM Estimation ------------------------
            # estimation["dm/score"].append(scores["score"])
            # estimation["dm/pred_reward_mean"].append(scores["pred_reward_mean"])
            # estimation["dm/pred_reward_total"].append(scores["pred_reward_total"])

            # IS Estimation -----------------------
            estimation["is/V_prev"].append(is_estimation["V_prev"])
            estimation["is/V_step_IS"].append(is_estimation["V_step_IS"])
            estimation["is/V_gain_est"].append(is_estimation["V_gain_est"])

    est_mean = pd.DataFrame.from_dict(estimation).mean(axis=0)
    custom_register_mean = pd.DataFrame.from_dict(custom_register).mean(axis=0)
    custom_deal_mean = pd.DataFrame.from_dict(custom_deal).mean(axis=0)
    custom_ltv_mean = pd.DataFrame.from_dict(custom_ltv).mean(axis=0)

    # Accuracy, Precision, Recall, F1
    true_actions = np.array(true_actions, dtype=np.float)
    pred_actions = np.array(actions, dtype=np.float)
    # print(actions)
    # print(pred_actions.shape)

    accuracy = (pred_actions == true_actions).sum() / len(true_actions)
    true_positives = ((pred_actions == 1) & (true_actions == 1)).sum()
    false_positives = ((pred_actions == 1) & (true_actions == 0)).sum()
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / true_actions.sum()
    f1 = (2 * precision * recall) / (precision + recall)

    # Confusion Matrix
    print(type(true_actions))
    print(type(pred_actions))
    cm = confusion_matrix(true_actions.round(), pred_actions.round())

    figure = utils.plot_confusion_matrix(cm, class_names=["Don't activate", "Activate"])
    # buf = io.BytesIO()
    # figure.savefig(buf)
    # buf.seek(0)
    # im = Image.open(buf)

    # height, width, channel = np.asarray(im).shape
    # image = Image.fromarray(np.asarray(im))

    # output = io.BytesIO()
    # image.save(output, format="PNG")
    # image_string = output.getvalue()
    # output.close()
    writer.add_figure("confusion_matrix", figure)

    writer.add_scalar(
        tag="Evaluation/is/V_prev",
        scalar_value=est_mean["is/V_prev"],
        global_step=outer_iter,
    )
    writer.add_scalar(
        tag="Evaluation/is/V_step_IS",
        scalar_value=est_mean["is/V_step_IS"],
        global_step=outer_iter,
    )
    writer.add_scalar(
        tag="Evaluation/is/V_gain_est",
        scalar_value=est_mean["is/V_gain_est"],
        global_step=outer_iter,
    )
    writer.add_scalar(
        tag="Evaluation/actions_prob",
        scalar_value=float(actions.count(1)) / len(actions),
        global_step=outer_iter,
    )
    writer.add_scalar(
        tag="Evaluation/register/is/V_prev",
        scalar_value=custom_register_mean["register/is/V_prev"],
        global_step=outer_iter,
    )
    writer.add_scalar(
        tag="Evaluation/register/is/V_step_IS",
        scalar_value=custom_register_mean["register/is/V_step_IS"],
        global_step=outer_iter,
    )
    writer.add_scalar(
        tag="Evaluation/register/is/V_gain_est",
        scalar_value=custom_register_mean["register/is/V_gain_est"],
        global_step=outer_iter,
    )
    writer.add_scalar(
        tag="Evaluation/deal/is/V_prev",
        scalar_value=custom_deal_mean["deal/is/V_prev"],
        global_step=outer_iter,
    )
    writer.add_scalar(
        tag="Evaluation/deal/is/V_step_IS",
        scalar_value=custom_deal_mean["deal/is/V_step_IS"],
        global_step=outer_iter,
    )
    writer.add_scalar(
        tag="Evaluation/deal/is/V_gain_est",
        scalar_value=custom_deal_mean["deal/is/V_gain_est"],
        global_step=outer_iter,
    )
    writer.add_scalar(
        tag="Evaluation/ltv/is/V_prev",
        scalar_value=custom_ltv_mean["ltv/is/V_prev"],
        global_step=outer_iter,
    )
    writer.add_scalar(
        tag="Evaluation/ltv/is/V_step_IS",
        scalar_value=custom_ltv_mean["ltv/is/V_step_IS"],
        global_step=outer_iter,
    )
    writer.add_scalar(
        tag="Evaluation/ltv/is/V_gain_est",
        scalar_value=custom_ltv_mean["ltv/is/V_gain_est"],
        global_step=outer_iter,
    )
    writer.add_scalar(
        tag="Evaluation/accuracy", scalar_value=accuracy, global_step=outer_iter
    )
    writer.add_scalar(
        tag="Evaluation/precision", scalar_value=precision, global_step=outer_iter
    )
    writer.add_scalar(
        tag="Evaluation/recall", scalar_value=recall, global_step=outer_iter
    )
    writer.add_scalar(tag="Evaluation/f1", scalar_value=f1, global_step=outer_iter)

    # --------------------------------------------------------------------------------

    eval_score = est_mean["is/V_gain_est"]
    logger.log_kv("eval_score", eval_score)

    # track best performing policy
    policy_score = eval_score
    if policy_score > best_perf:
        best_policy = copy.deepcopy(policy)  # safe as policy network is clamped to CPU
        best_perf = policy_score

    tf = timer.time()
    logger.log_kv("iter_time", tf - ts)
    for key in agent.logger.log.keys():
        logger.log_kv(key, agent.logger.log[key][-1])
    print_data = sorted(
        filter(
            lambda v: np.asarray(v[1]).size == 1, logger.get_current_log_print().items()
        )
    )
    print(tabulate(print_data))
    logger.save_log(OUT_DIR + "/logs")

    if outer_iter > 0 and outer_iter % job_data["save_freq"] == 0:
        # convert to CPU before pickling
        agent.to("cpu")
        # make observation mask part of policy for easy deployment in environment
        old_in_scale = policy.in_scale
        for pi in [policy, best_policy]:
            pi.set_transformations(in_scale=1.0 / e.obs_mask)
        pickle.dump(
            agent,
            open(OUT_DIR + "/iterations/agent_" + str(outer_iter) + ".pickle", "wb"),
        )
        pickle.dump(
            policy,
            open(OUT_DIR + "/iterations/policy_" + str(outer_iter) + ".pickle", "wb"),
        )
        pickle.dump(best_policy, open(OUT_DIR + "/iterations/best_policy.pickle", "wb"))
        agent.to(job_data["device"])
        for pi in [policy, best_policy]:
            pi.set_transformations(in_scale=old_in_scale)
        make_train_plots(
            log=logger.log,
            keys=["rollout_score", "eval_score", "rollout_metric", "eval_metric"],
            x_scale=float(job_data["act_repeat"]),
            y_scale=1.0,
            save_loc=OUT_DIR + "/logs/",
        )

# final save
pickle.dump(agent, open(OUT_DIR + "/iterations/agent_final.pickle", "wb"))
policy.set_transformations(in_scale=1.0 / e.obs_mask)
pickle.dump(policy, open(OUT_DIR + "/iterations/policy_final.pickle", "wb"))
