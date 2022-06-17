#!/usr/bin/env python3

import argparse
import os
import random

import cv2
import numpy as np
from flightgym import VisionEnv_v1
from rpg_baselines.torch.envs import vec_env_wrapper as wrapper
from ruamel.yaml import YAML, RoundTripDumper, dump


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--render", type=int, default=1, help="Render with Unity")
    parser.add_argument("--frame", type=int, default=1000, help="Frame")
    return parser


def main():
    args = parser().parse_args()

    # load configurations
    cfg = YAML().load(
        open(
            os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"
        )
    )

    if args.render:
        # to connect unity
        cfg["unity"]["render"] = "yes"
    else:
        cfg["unity"]["render"] = "no"


    cfg["environment"]["control_feedthrough"] = True

    # load the Unity standardalone, make sure you have downloaded it.
    os.system(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64 &")

    # define the number of environment for parallelization simulation
    cfg["simulation"]["num_envs"] = 1

    # create training environment
    env = VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    # print("env set is finished")
    env = wrapper.FlightEnvVec(env)
    # print("env wrapper is finished")

    ep_length = 1000

    obs_dim = env.obs_dim
    act_dim = env.act_dim
    print("act_dim is " + str(act_dim))
    num_env = env.num_envs

    print("input envs valiable")

    env.reset(random=True)
    print("reset env")

    # connect unity
    if args.render:
      env.connectUnity()

    for frame_id in range(ep_length):
      dummy_actions = np.array([[0,0,0.5,0,0,0,0]])

      env.step(dummy_actions)

      if args.render:
          env.render(frame_id = frame_id)

      cv2.waitKey(20)
      continue

    #
    if args.render:
        env.disconnectUnity()


if __name__ == "__main__":
    main()

