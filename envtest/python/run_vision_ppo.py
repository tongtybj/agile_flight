#!/usr/bin/env python3
import argparse
import math
#
import os

import numpy as np
import torch

import time
from flightgym import VisionEnv_v1
from ruamel.yaml import YAML, RoundTripDumper, dump
# from ruamel import yaml
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo.policies import MlpPolicy

from rpg_baselines.torch.common.ppo import PPO
from rpg_baselines.torch.envs import vec_env_wrapper as wrapper
from rpg_baselines.torch.common.util import test_policy


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--train", type=int, default=1, help="Train the policy or evaluate the policy")
    parser.add_argument("--render", type=int, default=0, help="Render with Unity")
    parser.add_argument("--trial", type=int, default=1, help="PPO trial number")
    parser.add_argument("--iter", type=int, default=100, help="PPO iter number")
    parser.add_argument("--move_coeff", type=float, help="move_coeff of rewards")
    parser.add_argument("--collision_coeff", type=float, help="collision_coeff of rewards")
    parser.add_argument("--collision_exp_coeff", type=float, help="collision_exp_coeff of rewards")
    parser.add_argument("--survive_rew", type=float, help="collision_coeff of rewards")
    parser.add_argument("--check", type=bool, default = False, help="check of simulation, not make long csv")
    return parser


def main():
    # start_time = time.time()
    args = parser().parse_args()

    # load configurations
    cfg = YAML().load(
        open(
            os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"
        )
    )

    if not args.train:
        cfg["simulation"]["num_envs"] = 1 

    # print(args.move_coeff)
    if args.move_coeff != None:
        cfg["rewards"]["move_coeff"] = args.move_coeff
    if args.collision_coeff != None:
        cfg["rewards"]["collision_coeff"] = args.collision_coeff
    if args.collision_exp_coeff != None:
        cfg["rewards"]["collision_exp_coeff"] = args.collision_exp_coeff
    if args.survive_rew != None:
        cfg["rewards"]["survive_rew"] = args.survive_rew
    # print(cfg["rewards"]["move_coeff"])

    # create training environment
    train_env = VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.FlightEnvVec(train_env)

    # set random seed
    configure_random_seed(args.seed, env=train_env)

    if args.render:
        cfg["unity"]["render"] = "yes"
    
    # create evaluation environment
    old_num_envs = cfg["simulation"]["num_envs"]
    cfg["simulation"]["num_envs"] = 1
    eval_env = wrapper.FlightEnvVec(
        VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    )
    cfg["simulation"]["num_envs"] = old_num_envs

    # save the configuration and other files
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/saved"
    os.makedirs(log_dir, exist_ok=True)
    if args.train:
        model = PPO(
            tensorboard_log=log_dir,
            policy="MlpPolicy",
            policy_kwargs=dict(
                activation_fn=torch.nn.ReLU,
                net_arch=[dict(pi=[256, 256], vf=[512, 512])],
                log_std_init=-0.5,
            ),
            env=train_env,
            eval_env=eval_env,
            use_tanh_act=True,
            gae_lambda=0.95,
            gamma=0.99,
            n_steps=250,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            batch_size=25000,
            clip_range=0.2,
            use_sde=False,  # don't use (gSDE), doesn't work
            env_cfg=cfg,
            verbose=1,
            check = args.check
        )
        # print(model.logger)
        model.learn(total_timesteps=int(25E5), log_interval=(10, 50))
        cfg_dir = model.logger.get_dir()+"/config_new.yaml"
        with open(cfg_dir, "w") as outfile:
            dump({
        "rewards": {
            "move_coeff": args.move_coeff,
            "collision_coeff": args.collision_coeff,
            "collision_exp_coeff": args.collision_exp_coeff,
            "survive_rew": args.survive_rew,
        }
    }, outfile, default_flow_style=False)
        # finish_time = time.time()
        # print("learning time is "+ str(finish_time-start_time))
    else:
        os.system(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64 &")
        #
        weight = rsg_root + "/saved/PPO_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)
        env_rms = rsg_root +"/saved/PPO_{0}/RMS/iter_{1:05d}.npz".format(args.trial, args.iter)

        device = get_device("auto")
        saved_variables = torch.load(weight, map_location=device)
        # Create policy object
        policy = MlpPolicy(**saved_variables["data"])
        #
        policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
        # Load weights
        policy.load_state_dict(saved_variables["state_dict"], strict=False)
        policy.to(device)
        # 
        eval_env.load_rms(env_rms)
        test_policy(eval_env, policy, render=args.render)



if __name__ == "__main__":
    main()
