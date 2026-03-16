"""ADAS inference runner: rollout + scoring -> CSV.

Reuses verl's Ray + vLLM rollout + reward function infrastructure,
but skips all training components (optimizer, critic, ref policy).

Output CSV (token, pdms, pdms_scaled) is compatible with the existing
ADAS pipeline (run_adas.sh -> merge -> stats -> filter).

Usage:
    python -m verl.trainer.main_adas \
        config=examples/config_vla.yaml \
        data.train_files=<full_dataset>@train \
        data.val_files=<full_dataset>@test \
        worker.actor.model.model_path=<model_path> \
        worker.rollout.n=8 \
        worker.reward.reward_function=<reward_fn> \
        trainer.experiment_name=<exp_name> \
        trainer.n_gpus_per_node=8
"""

import csv
import json
import os

import ray
from omegaconf import OmegaConf
from ray.experimental.tqdm_ray import tqdm

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import ResourcePoolManager, Role


@ray.remote(num_cpus=1)
class AdasRunner:
    """Lightweight runner: vLLM rollout + reward scoring only."""

    def run(self, config: PPOConfig) -> tuple[str, int]:
        """Run ADAS inference and return (output_dir, n_rollout)."""
        print(json.dumps(config.to_dict(), indent=2))

        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # --- Worker setup (ActorRolloutRef only, no critic) ---
        role_worker_mapping = {Role.ActorRolloutRef: ray.remote(FSDPWorker)}
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {Role.ActorRolloutRef: global_pool_id}
        rpm = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        rpm.create_resource_pool()

        resource_pool = rpm.get_resource_pool(Role.ActorRolloutRef)
        actor_cls = RayClassWithInitArgs(
            cls=role_worker_mapping[Role.ActorRolloutRef],
            config=config.worker,
            role="actor_rollout_ref",
        )
        worker_dict_cls = create_colocated_worker_cls(class_dict={"actor_rollout_ref": actor_cls})
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        all_wg = wg.spawn(prefix_set={"actor_rollout_ref"})
        actor_wg = all_wg["actor_rollout_ref"]
        actor_wg.init_model()

        # --- Load checkpoint (if specified) ---
        load_path = config.trainer.load_checkpoint_path
        if load_path is None and config.trainer.find_last_checkpoint:
            from ..utils.checkpoint import find_latest_ckpt
            load_path, _ = find_latest_ckpt(config.trainer.save_checkpoint_path)

        if load_path is not None:
            actor_path = os.path.join(load_path, "actor")
            print(f"Loading checkpoint: {actor_path}")
            actor_wg.load_checkpoint(actor_path)

        # --- Reward function ---
        RemoteRewardManager = ray.remote(AutoRewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

        # --- Dataloader (reuse create_dataloader; val is unused but kept for similarity with main.py) ---
        train_dataloader, _ = create_dataloader(config.data, tokenizer, processor)

        # --- Output setup ---
        output_dir = os.path.join("checkpoints", "adas", config.trainer.experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "adas_scores.csv")

        n = config.worker.rollout.n
        total_rows = 0
        num_batches = len(train_dataloader)
        print(f"ADAS inference: n={n}, output={csv_path}")
        print(f"Train dataloader: {num_batches} batches")

        # --- Inference + scoring loop ---
        actor_wg.prepare_rollout_engine()

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["token", "pdms", "pdms_scaled"])
            writer.writeheader()

            for batch_idx, batch_dict in enumerate(tqdm(train_dataloader, total=num_batches, desc="ADAS inference")):
                batch = DataProto.from_single_dict(batch_dict, meta_info={
                    "min_pixels": config.data.min_pixels,
                    "max_pixels": config.data.max_pixels,
                    "video_fps": config.data.video_fps,
                })
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
                )

                gen_batch, pad_size = pad_dataproto_to_divisor(gen_batch, actor_wg.world_size)
                gen_output = actor_wg.generate_sequences(gen_batch)
                gen_output = unpad_dataproto(gen_output, pad_size=pad_size * n)

                batch = batch.repeat(repeat_times=n, interleave=True)
                batch = batch.union(gen_output)

                reward_tensor, reward_metrics = ray.get(reward_fn.compute_reward.remote(batch))

                gt_list = batch.non_tensor_batch["ground_truth"]
                pdms_list = reward_metrics.get("pdms", [])
                accuracy_list = reward_metrics.get("accuracy", [])

                for i, gt in enumerate(gt_list):
                    gt_obj = gt if isinstance(gt, dict) else json.loads(gt)
                    writer.writerow({
                        "token": gt_obj["token"],
                        "pdms": pdms_list[i] if i < len(pdms_list) else 0.0,
                        "pdms_scaled": accuracy_list[i] if i < len(accuracy_list) else 0.0,
                    })
                    total_rows += 1


        actor_wg.release_rollout_engine()
        print(f"ADAS scores written: {csv_path} ({total_rows} rows)")

        return output_dir, n


def main():
    cli_args = OmegaConf.from_cli()

    default_config = OmegaConf.structured(PPOConfig())
    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)

    # Force settings for inference-only mode
    ppo_config.algorithm.adv_estimator = "grpo"
    ppo_config.algorithm.disable_kl = True
    ppo_config.data.token_filter_file = None
    ppo_config.data.shuffle = False

    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            }
        }
        ray.init(runtime_env=runtime_env)

    runner = AdasRunner.remote()
    output_dir, n = ray.get(runner.run.remote(ppo_config))

    print(f"\nADAS inference complete. CSV at: {output_dir}/adas_scores.csv")
    ray.shutdown()


if __name__ == "__main__":
    main()
