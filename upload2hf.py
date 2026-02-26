from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your model files
upload_folder(folder_path="./checkpoints/easy_r1/qwen2_5_vl_3b_navsim_adas_3x_1k/global_step_150/actor", 
              repo_id="MashiroLn/Curious-VLA", 
              repo_type="model")

# TOKEN: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