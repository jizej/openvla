# VLA Scripts Guide

Short pointers for the scripts under `vla-scripts/`, what they do, and the
minimum CLI you need to run them. All commands assume you are in the repo root.

## Core training and evaluation
- `finetune.py` — LoRA fine-tuning of an OpenVLA model on an RLDS-style dataset.
  Typical multi-GPU run:
  ```
  torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir /media/data \
    --dataset_name cereal \
    --run_root_dir /home/jizej/Workspaces/cs498-robot/project/openvla/checkpoints \
    --lora_rank 32 --batch_size 8 --grad_accumulation_steps 2 \
    --learning_rate 5e-4 --image_aug False --save_steps 250 --epochs 5
  ```
  You can point `--vla_path` at an existing LoRA run to continue training or to
  fine-tune a merged checkpoint.

- `eval.py` — offline evaluation on a dataset; mirrors the finetune config but
  does not update weights. Example:
  ```
  torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/eval.py \
    --vla_path checkpoints/<run_dir> \
    --data_root_dir /media/data \
    --dataset_name my_cereal_task_trajectories
  ```

## Dataset generation and visualization (Robosuite)
- `generate_vla_dataset_trajectories.py` — headless Robosuite rollouts that
  build `episode_*` folders with RGB, depth, instruction, and affordance
  waypoints. Set `MUJOCO_GL=egl` for headless rendering.
- `generate_vla_dataset_visualize.py` — similar generator that also records
  camera intrinsics/extrinsics; used for visualization/debug.
- `visualize_dataset_and_affordances.py` — overlays affordance waypoints on RGB
  frames to sanity-check a generated dataset:
  ```
  python vla-scripts/visualize_dataset_and_affordances.py \
    --dataset_dir /path/to/episodes --output_dir /tmp/vis --num_samples 5
  ```

## Robosuite rollouts and waypoint evaluation
- `eval_robosuite.py` — runs OpenVLA inside Robosuite with on-the-fly waypoint
  generation; can save videos.
- `eval_waypoints.py` — **Main eval program**: runs a waypoint-conditioned policy in Robosuite and
  records videos/results (expects `PickPlaceClutter` env).
- `eval_waypoints_visualize.py` — variant that regenerates data with
  `generate_vla_dataset_visualize.py` and saves visual overlays.

## Deployment
- `deploy.py` — REST API server for inference. Start on GPU host:
  ```
  python vla-scripts/deploy.py --openvla_path checkpoints/<merged_or_base> \
    --stats_path checkpoints/<merged_or_base>/dataset_statistics.json \
    --predict_waypoint -1
  ```
  Then POST `{"image": <uint8 HxWx3>, "instruction": "..."}`
  to `http://<host>:8000/act`.
- `deploy_all_in_one.py` — same API with multi-waypoint support (pregrasp/grasp/
  release) from a single process; pass the per-waypoint model/stat paths.

## Example commands seen in this project
- Apptainer + multi-GPU fine-tune:
  ```
  apptainer exec --bind /media/data --bind /home/jizej/Workspaces/cs498-robot/project/openvla/ --nv containers/openvla_app_fix.sif \
    torchrun --standalone --nnodes 1 --nproc-per-node 2 openvla/vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b --data_root_dir /media/data --dataset_name cereal \
    --run_root_dir /home/jizej/Workspaces/cs498-robot/project/openvla/checkpoints \
    --lora_rank 32 --batch_size 8 --grad_accumulation_steps 2 --learning_rate 5e-4 \
    --image_aug False --save_steps 250 --epochs 5
  ```
- Continue fine-tuning a prior run:
  ```
  torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
    --vla_path checkpoints/openvla-7b+my_cereal_task_trajectories+b20+lr-0.0005+lora-r32+dropout-0.0+251211_0203 \
    --data_root_dir /media/data --dataset_name my_cereal_task_trajectories \
    --run_root_dir /home/jizej/Workspaces/cs498-robot/project/openvla/checkpoints \
    --lora_rank 32 --batch_size 10 --grad_accumulation_steps 2 --learning_rate 5e-4 \
    --image_aug False --save_steps 250 --epochs 1
  ```
- Evaluate a checkpoint:
  ```
  torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/eval.py \
    --vla_path checkpoints/openvla-7b+my_cereal_task_trajectories+b20+lr-0.0005+lora-r32+dropout-0.0+251212_0455 \
    --data_root_dir /media/data --dataset_name my_cereal_task_trajectories
  ```
- Quick dataset/affordance visualization:
  ```
  python vla-scripts/visualize_dataset_and_affordances.py --dataset_dir /media/data/my_data --output_dir ./my_data_VISUAL
  ```

## Notes
- Most scripts expect a CUDA GPU. Set `MUJOCO_GL=egl` for headless Robosuite.
- `--predict_waypoint` controls whether the model predicts full actions (-1) or
  specific waypoints (0=pregrasp, 1=grasp, 2=release).
- Robosuite assets live under `vla-scripts/robosuite/`; ensure `pick_place_clutter`
  is importable before running Robosuite-based scripts.
