python pipeline.py \
    --infer_folder /mnt/data/ccy/VLA_train/parallel_infer/output/4B_navsim_prn103kv4_norm_stage2_pn103kv3_use_refinev4_norm_stage1 \
    --csv_mode prefill \
    -p 0.1 \
    --conf 0.1 \
    --group_size_mode from_csv \
    --target_min 3000 \
    --target_max 6000