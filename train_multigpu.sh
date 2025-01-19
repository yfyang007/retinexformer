config=$1
gpu_ids=$2
master_port=${3:-4321} # Please use different master_port for different training processes.

gpu_count=$(echo $gpu_ids | tr -cd ',' | wc -c)
gpu_count=$((gpu_count + 1))

# pytorch1.x
CUDA_VISIBLE_DEVICES=$gpu_ids python -m torch.distributed.launch --nproc_per_node=$gpu_count --master_port=$master_port basicsr/train.py --opt $config --launcher pytorch

# pytorch2.x
# CUDA_VISIBLE_DEVICES=$gpu_ids torchrun --nproc_per_node=$gpu_count --master_port=$master_port basicsr/train.py --opt $config --launcher pytorch
