for theta in $(seq 0.1 0.1 0.9);
do
    python main.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --seed 0 --batch_size 512 --model resnet50 --max_epoch 1000 --gamma 0.25 --theta ${theta} --loss wCE --dont_set_seed 1 --gpu_num 0 --log_dir 'result/water_birds/ours/theta='${theta}'/'
done