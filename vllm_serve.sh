pkill -9 python3.10
pkill -9 vllm

# model_name_or_path=/chencanyu-sh2/VLA_train/output/train/InternVL3/real_2B_navtrain_recog/v4-20250902-124247/checkpoint-249/
model_name_or_path=/mnt/data/ccy/EasyR1/checkpoints/easy_r1/qwen2_5_vl_3b_navsim_grpo_balanced_analysis/global_step_75/actor/huggingface
# model_name_or_path=/chencanyu-sh2/huggingface_hub/ReCogDrive/ReCogDrive_VLM_2B
# WORLD_SIZE=1 RANK=0 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3     vllm serve $model_name_or_path --host 0.0.0.0 --port 8192 --tensor-parallel-size 4 &
# WORLD_SIZE=1 RANK=0 ASCEND_RT_VISIBLE_DEVICES=4,5,6,7     vllm serve $model_name_or_path --host 0.0.0.0 --port 8193 --tensor-parallel-size 4 &
# WORLD_SIZE=1 RANK=0 ASCEND_RT_VISIBLE_DEVICES=8,9,10,11   vllm serve $model_name_or_path --host 0.0.0.0 --port 8194 --tensor-parallel-size 4 &
# WORLD_SIZE=1 RANK=0 ASCEND_RT_VISIBLE_DEVICES=12,13,14,15 vllm serve $model_name_or_path --host 0.0.0.0 --port 8195 --tensor-parallel-size 4

num_instances=8
start_port=9515

if [ -z "$num_instances" ]; then
  echo "please provide num_instances: $0 4"
  exit 1
fi

cards_per_instance=$((8 / num_instances))

for i in $(seq 0 $((num_instances-1))); do
  start_card=$((i * cards_per_instance))
  end_card=$((start_card + cards_per_instance - 1))
  port=$((start_port + i))

  devices=$(seq -s, $start_card $end_card)

  echo "Start Instance $i: Use Devices: $devices, Port: $port"

  CUDA_VISIBLE_DEVICES=$devices vllm serve \
    $model_name_or_path \
    --host 0.0.0.0 \
    --port $port &
done