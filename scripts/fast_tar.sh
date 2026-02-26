exp_name=qwen3_vl_4b_openvit_refine_prn103k_v3tov4_3k_sdr
step=50


model_path=/mnt/data/ccy/EasyR1/checkpoints/easy_r1/$exp_name/global_step_$step/actor/huggingface
output_path=/mnt/data/ccy/EasyR1/checkpoints/easy_r1/$exp_name/
output_name=${exp_name}_step_${step}.tar

tar -cvf $output_path/$output_name -C $model_path .

oss cp $output_path/$output_name oss://weights/$output_name