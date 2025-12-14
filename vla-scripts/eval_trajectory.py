import os
import sys
import time
import json
import pprint
import signal
import datetime
import threading

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from copy import deepcopy

import draccus
import torch
import torch.distributed as dist
import numpy as np
import tqdm
import contextlib

from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics, AttributeDict
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS, rlds_dataset_builder_transform

from prismatic.vla.sim.mimicgen import MGStreamingDataset
from prismatic.util.grokfast import gradfilter_ma, gradfilter_ema

from merge import merge_lora


# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class EvalConfig:
    # fmt: off
    exp_id: str = None                                              # Unique experiment ID (will be initialized if left None)
    exp_tag: str = None                                             # Extra tag to end onto the end of experiment ID string
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    output_dir: Path = Path("eval_output")                               # Path to directory to store model output

    # Evaluation Parameters
    batch_size: int = 1                                           # Fine-tuning batch size
    metric_steps: int = 4                                          # The number of batches to average loss/accuracy metrics over
    grad_accumulation_steps: int = 2                                # Gradient accumulation steps

    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    image_aug: bool = False                                          # Whether to train with image augmentations

    predict_waypoint: int = -1                                        # -1: predict action, 0: predict pregrasp, 1: predict grasp, 2: predict release
    # fmt: on

@draccus.wrap()
def eval(cfg: EvalConfig) -> None:
    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    #torch.distributed.init_process_group(backend='gloo')
    distributed_state = PartialState(backend='gloo')
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    print(f"Evaluating OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")


    # TODO: replace vla_path with vla_checkpoint (merged with lora)
    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        quantization_config=None,  # no quantization for now
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    vla = vla.to(device_id)
    vla.eval()
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Evaluation Dataset
    if cfg.dataset_name not in OXE_DATASET_CONFIGS:
        data_cfg = deepcopy(OXE_DATASET_CONFIGS['rlds_dataset_builder'])
        OXE_DATASET_CONFIGS[cfg.dataset_name] = data_cfg

    if cfg.dataset_name not in OXE_STANDARDIZATION_TRANSFORMS:
        OXE_STANDARDIZATION_TRANSFORMS[cfg.dataset_name] = rlds_dataset_builder_transform

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        predict_waypoint=cfg.predict_waypoint
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        train=False,  # important, TODO: check eval set loading logic
    )


    # # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    # if distributed_state.is_main_process:
    #     save_dataset_statistics(vla_dataset.dataset_statistics, cfg.run_root_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    metrics = AttributeDict(
        loss_action = Metric('loss_action', 'Loss/action'),
        token_accuracy = Metric('token_accuracy', 'Accuracy/tokens'),
        action_accuracy = Metric('action_accuracy', 'Accuracy/action'),
    )

    for metric in metrics.values():
        metric.resize(cfg.metric_steps, cfg.grad_accumulation_steps)
    cfg.max_steps = int(len(vla_dataset) / cfg.batch_size / cfg.grad_accumulation_steps) #int(len(dataloader) * cfg.epochs / cfg.grad_accumulation_steps)


    # Allow the user to interrupt training with Ctrl+C
    interrupts = ProcessInterrupt()
    time_begin = time.perf_counter()
    # Keep filling data until the requested number of steps is reached
    def next_batch():
        batch_idx = 0
        while True:
            for batch in dataloader:
                if batch_idx / cfg.grad_accumulation_steps >= cfg.max_steps:
                    return
                yield batch_idx, batch
                batch_idx += 1

    # Train!
    with tqdm.tqdm(total=500, leave=False) as progress:
        vla.eval()

        for batch_idx, batch in next_batch():

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            metrics.token_accuracy += correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()))
            continuous_actions_gt = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()))

            metrics.loss_action += torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
            metrics.action_accuracy += 1.0 - metrics.loss_action[-1]


            progress.set_description(f"{metrics.token_accuracy}  {metrics.action_accuracy}")
            progress.update() # increments progress.n (global step count)


            # save metrics
            save_metrics(metrics, batch_idx, cfg)

            # Save model output as numpy objects 
            output_dir = getattr(cfg, 'output_dir', None)
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                batch_identifier = f"step_{batch_idx}"

                npz_files = {}
                # Save each output as an individual numpy array file
                for key, value in [
                    ("action_logits", action_logits.detach().cpu().numpy()),
                    ("action_preds", action_preds.detach().cpu().numpy()),
                    ("action_gt", action_gt.detach().cpu().numpy()),
                    ("continuous_actions_pred", continuous_actions_pred.detach().cpu().numpy()),
                    ("continuous_actions_gt", continuous_actions_gt.detach().cpu().numpy()),
                ]:
                    npy_path = os.path.join(output_dir, f"{batch_identifier}_{key}.npy")
                    np.save(npy_path, value)
                    npz_files[key] = npy_path
            
            if progress.n >= 500:
                break
                

        # print evaluation stats
        eval_time = time.perf_counter() - time_begin
        eval_frames = progress.n * cfg.batch_size * cfg.grad_accumulation_steps
        eval_rate = eval_frames / eval_time

        print(f"\nDone evaluating after {progress.n} steps, {eval_frames} frames  ({int(eval_time)} seconds, {eval_rate:.2f} fps)")


def save_metrics(metrics, steps, cfg):
    stats_path = cfg.output_dir / "eval_statistics.json"
    stats_past = [] 

    try:
        with open(stats_path, 'r') as f:
            stats_past = json.load(f)
    except Exception:
        pass

    stats = {
        'steps': steps,
    }

    for metric in metrics.values():
        stats[metric.name] = {
            'step': metric.step_mean(),
            'mean': metric.mean(),
        }

    try:
        with open(stats_path, 'w') as f:
            json.dump(stats_past + [stats], f, indent=2)
    except Exception as error:
        print(f"Exception while saving training stats to {stats_path}\n  {error}")

    print(f"Saving training stats to {stats_path}")

class Metric:
    # Accumulate / average training stats
    def __init__(self, name, tensorboard=None, window=10, step_window=None):
        self.name = name
        self.tensorboard = tensorboard
        self.resize(window, step_window)

    def __str__(self):
        return f"{self.name}={self.step_mean():.4f} ~{self.mean():.4f}"

    def __len__(self):
        return len(self.history)

    def __getitem__(self, index):
        return self.history[index]

    def __iadd__(self, value):
        self.append(value)
        return self

    def append(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.history.append(value)

    def mean(self, window=None):
        if window and window <= len(self.history):
            history = [self.history[-x-1] for x in range(window)] #self.history[-window:]
        else:
            history = self.history
        return sum(history) / len(history)

    def step_mean(self):
        return self.mean(self.step_window)

    def resize(self, window=10, step_window=None):
        self.history = deque(maxlen=window)
        self.window = window
        self.step_window = step_window


class ProcessInterrupt(threading.Thread):
    # Ctrl+D interrupt handler
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interrupts = 0
        self.start()

    def __len__(self):
        return self.interrupts

    def run(self):
        print(">> Press Ctrl+D to save weights and stop training early\n")
        while True:
            try:
                input()
            except EOFError:
                self.interrupts += 1
                if self.interrupts > 2:
                    print("\nTerminating training process early\n")
                    sys.exit(0)
                elif self.interrupts > 1:
                    print("\nPress Ctrl+D again for ungraceful termination\n")
                else:
                    print("\nCtrl+D pressed, interrupting training...\n")


if __name__ == "__main__":
    eval()