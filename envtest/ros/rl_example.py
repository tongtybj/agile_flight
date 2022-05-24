#!/usr/bin/python3

import os

import torch
import numpy as np

# 
from ruamel.yaml import YAML
from utils import AgileCommand
from scipy.spatial.transform import Rotation as R

# stable baselines 
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo.policies import MlpPolicy

def normalize_obs(obs, obs_mean, obs_var):
    return (obs - obs_mean) / np.sqrt(obs_var + 1e-8)

def rl_example(state, obstacles, rl_policy=None):
    policy, obs_mean, obs_var, act_mean, act_std = rl_policy
    # Convert obstacles to vector observation
    # obs_vec = []
    # for obstacle in obstacles.obstacles:
    #     obs_vec.append(obstacle.position.x)
    #     obs_vec.append(obstacle.position.y)
    #     obs_vec.append(obstacle.position.z)
    #     obs_vec.append(obstacle.scale)
    obs_vec = np.array(obstacles.boxel)

    # Convert state to vector observation
    goal_vel = np.array([5.0, 0.0, 0.0]) 
    world_box = np.array([-20, 80, -10, 10, 0.0, 10])

    att_aray = np.array([state.att[1], state.att[2], state.att[3], state.att[0]])
    rotation_matrix = R.from_quat(att_aray).as_matrix().reshape((9,), order="F")
    obs = np.concatenate([
        goal_vel, rotation_matrix, state.pos, state.vel, obs_vec, 
        np.array([world_box[2] - state.pos[1], world_box[3] - state.pos[1], 
        world_box[4] - state.pos[2] , world_box[5] - state.pos[2]]),
  state.omega], axis=0).astype(np.float64)

    obs = obs.reshape(-1, obs.shape[0])
    norm_obs = normalize_obs(obs, obs_mean, obs_var)
    #  compute action
    action, _ = policy.predict(norm_obs, deterministic=True)
    action = (action * act_std + act_mean)[0, :]

    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = action[0] 
    command.bodyrates = action[1:4] 
    return command

def rl_example_vision(state, img, rl_policy=None):
    policy, obs_mean, obs_var, act_mean, act_std = rl_policy
    # Convert obstacles to vector observation
    # obs_vec = []
    # for obstacle in obstacles.obstacles:
    #     obs_vec.append(obstacle.position.x)
    #     obs_vec.append(obstacle.position.y)
    #     obs_vec.append(obstacle.position.z)
    #     obs_vec.append(obstacle.scale)
    obs_vec = img_to_obs(img)

    # Convert state to vector observation
    goal_vel = np.array([5.0, 0.0, 0.0]) 
    world_box = np.array([-20, 80, -10, 10, 0.0, 10])

    att_aray = np.array([state.att[1], state.att[2], state.att[3], state.att[0]])
    rotation_matrix = R.from_quat(att_aray).as_matrix().reshape((9,), order="F")
    obs = np.concatenate([
        goal_vel, rotation_matrix, state.pos, state.vel, obs_vec, 
        np.array([world_box[2] - state.pos[1], world_box[3] - state.pos[1], 
        world_box[4] - state.pos[2] , world_box[5] - state.pos[2]]),
  state.omega], axis=0).astype(np.float64)

    obs = obs.reshape(-1, obs.shape[0])
    norm_obs = normalize_obs(obs, obs_mean, obs_var)
    #  compute action
    action, _ = policy.predict(norm_obs, deterministic=True)
    action = (action * act_std + act_mean)[0, :]

    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = action[0] 
    command.bodyrates = action[1:4] 
    return command

def img_to_obs(img): #http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html #change hyperparameters!
    print("change image to obs")
    env_cuts = 8
    boundary = int(env_cuts/2)
    obstacle_obs = np.zeros(env_cuts*env_cuts)
    for theta in range(-boundary, boundary):
        for phi in range(-boundary, boundary):
            tcell = (t+0.5)*(PI/env_cuts)/2
            pcell = (f+0.5)*(PI/env_cuts)/2
            obstacle_obs[(t+boundary)*env_cuts+(f+boundary)] = getClosestDistance(img.data, tcell,pcell)



def load_rl_policy(policy_path):
    print("============ policy_path: ", policy_path)
    policy_dir = policy_path  + "/policy.pth"
    rms_dir = policy_path + "/rms.npz"
    cfg_dir =  policy_path + "/config.yaml"

    # action 
    env_cfg = YAML().load(open(cfg_dir, "r"))
    quad_mass = env_cfg["quadrotor_dynamics"]["mass"]
    omega_max = env_cfg["quadrotor_dynamics"]["omega_max"]
    thrust_max = 4 * env_cfg["quadrotor_dynamics"]["thrust_map"][0] * \
        env_cfg["quadrotor_dynamics"]["motor_omega_max"] * \
        env_cfg["quadrotor_dynamics"]["motor_omega_max"]
    act_mean = np.array([thrust_max / quad_mass / 2, 0.0, 0.0, 0.0])[np.newaxis, :] 
    act_std = np.array([thrust_max / quad_mass / 2, \
       omega_max[0], omega_max[1], omega_max[2]])[np.newaxis, :] 

    rms_data = np.load(rms_dir)
    obs_mean = np.mean(rms_data["mean"], axis=0)
    obs_var = np.mean(rms_data["var"], axis=0)

    # # -- load saved varaiables 
    device = get_device("auto")
    saved_variables = torch.load(policy_dir, map_location=device)
    # Create policy object
    policy = MlpPolicy(**saved_variables["data"])
    #
    policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
    # Load weights
    policy.load_state_dict(saved_variables["state_dict"], strict=False)
    policy.to(device)

    return policy, obs_mean, obs_var, act_mean, act_std
