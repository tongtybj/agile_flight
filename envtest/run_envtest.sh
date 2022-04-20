#!/bin/sh
# python3 -m python.run_vision_ppo --render 0 --train 1 --move_coeff 0.5

for move_coeff in 0.5 1
do 
for collision_coeff in -0.1 -0.4 -1 -5
do
for survive_rew in 0.03 0.1 0.3
do
python3 -m python.run_vision_ppo --render 0 --train 1 --move_coeff $move_coeff --collision_coeff $collision_coeff --survive_rew $survive_rew
done
done
done