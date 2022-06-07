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
        # to simulate rgb camera
        cfg["rgb_camera"]["on"] = "yes"


    # load the Unity standardalone, make sure you have downloaded it.
    os.system(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64 &")

    # define the number of environment for parallelization simulation
    cfg["simulation"]["num_envs"] = 1 

    # print("dumper")
    # print(dump(cfg, Dumper=RoundTripDumper))
    # print("dumper finish")


    # create training environment
    env = VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    # print("env set is finished")
    env = wrapper.FlightEnvVec(env)
    # print("env wrapper is finished")

    ep_length = 100

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
    # print("connect unity")


    for frame_id in range(ep_length):
      print("Simuation step: {0}".format(frame_id))
      # generate dummmy action [-1, 1]
    #   dummy_actions = np.random.rand(num_env, act_dim) * 2 - np.ones(shape=(num_env, act_dim))
    #   print(dummy_actions)
      dummy_actions = np.array([[0,0,0.5,0.5,0.5,0.5,0.5]])
      print(dummy_actions)

      # A standard OpenAI gym style interface for reinforcement learning.    
      obs, rew, done, info = env.step(dummy_actions)
 
      #
      receive_frame_id = env.render(frame_id = frame_id)
      print('')
      print("sending frame id: ", frame_id, "received frame id: ", receive_frame_id)

      # ====== Retrive RGB Image From the simulator=========
      raw_rgb_img =env.getImage(rgb=True) 

      num_img = raw_rgb_img.shape[0] 
      num_col = 1
      num_row = int(num_img / num_col)

      rgb_img_list = []
      for col in range(num_col):
        rgb_img_list.append([])
        for row in range(num_row):
          rgb_img = np.reshape(
              raw_rgb_img[col*num_row + row], (env.img_height, env.img_width, 3))
          rgb_img_list[col] += [rgb_img]
      
      rgb_img_tile = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in rgb_img_list])
      cv2.imshow("rgb_img", rgb_img_tile)
      # cv2.imwrite("./images/img_{0:05d}.png".format(frame_id), rgb_img_tile)
      # wait for the purpose of using open cv visualization
      cv2.waitKey(500)

      # ======Retrive Depth Image=========
      raw_depth_images = env.getDepthImage()
      depth_img_list = []
      for col in range(num_col):
        depth_img_list.append([])
        for row in range(num_row):
          depth_img = np.reshape(
              raw_depth_images[col*num_row + row], (env.img_height, env.img_width))
          depth_img_list[col] += [depth_img]
      
      depth_img_tile = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in depth_img_list])
      cv2.imshow("depth_img", depth_img_tile)
      # wait for the purpose of using open cv visualization
      cv2.waitKey(500)

    #
    if args.render:
        env.disconnectUnity()


if __name__ == "__main__":
    main()

