#!/bin/sh

# for move_coeff in 0.5 1
# do 
# for collision_coeff in -0.1 -0.4 -1 -5
# do
# for survive_rew in 0.03 0.1 0.3
# do
# python3 -m python.run_vision_ppo --render 0 --train 1 --move_coeff $move_coeff --collision_coeff $collision_coeff --survive_rew $survive_rew
# done
# done
# done
start_time=`date +%s`
for move_coeff in 0.8 1.0 1.5 3
do 
for collision_coeff in -0.05 -0.1
do
for survive_rew in 0.1
do
for collision_exp_coeff in 0.1 0.3 1
do
python3 -m python.run_vision_ppo --render 0 --train 1 \
--move_coeff $move_coeff --collision_coeff $collision_coeff \
--collision_exp_coeff $collision_exp_coeff --survive_rew $survive_rew
# --check True
done
done
done
done
end_time=`date +%s`
run_time=$((end_time - start_time))
echo $run_time