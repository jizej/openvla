import robosuite as suite
from robosuite.utils import camera_utils
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import cv2

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

class Experiment:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

        # 1. Load Controller Config
        controller_name = "basic_abs_pose.json"
        controller_path = os.path.join(os.path.dirname(__file__), 'controllers', controller_name)
        
        if not os.path.exists(controller_path):
            print("=" * 80)
            print(f"FATAL ERROR: Controller file not found.")
            print(f"Expected path: {controller_path}")
            print(f"Please create a folder named 'controllers' and place '{controller_name}' inside it.")
            print("=" * 80)
            exit()
            
        print(f"Loading controller config from: {controller_path}")
        controller_config = suite.load_composite_controller_config(controller=controller_path)

        # 2. Create argument configuration
        self.config = {
            "env_name": "PickPlaceClutter",
            "robots": "Panda",
            "controller_configs": controller_config,
        }
        # 3. creat env
        print("Creating 'PickPlaceClutter' environment...")
        self.env = suite.make(
            **self.config,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
            camera_names="agentview",
            camera_heights=256,
            camera_widths=256,
            camera_depths=True,
            control_freq=20,
        )

        # init data and model
        



    def _video_frame(self, text=None):
        if not getattr(self, "_rec", None) or not self._rec["on"]:
            return

        H, W = self._rec["H"], self._rec["W"]
        cam = self._rec["camera"]

        # Render
        rgb = self.env.sim.render(camera_name=cam, height=H, width=W, depth=False)
        frame = cv2.flip(rgb, 0)

        # Ensure dtype + contiguity for VideoWriter
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

        # Optional overlays
        if text:
            cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        if self.trial_info_text:
            y = H - 10
            cv2.putText(frame, self.trial_info_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, self.trial_info_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bgr = np.ascontiguousarray(bgr)

        # Track frames-written to catch “empty” files
        self._rec["frames"] += 1
        self._rec["writer"].write(bgr)
        
    def _video_start(self, path="Q4.mp4", fps=30, H=256, W=256, camera_name="birdview"):
        self._rec = {"on": False, "path": path, "fps": fps, "H": H, "W": W, "camera": camera_name, "frames": 0}
        for fourcc_str in ("avc1", "mp4v", "XVID"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(path, fourcc, fps, (W, H), True)
            if writer.isOpened():
                self._rec.update({"writer": writer, "fourcc": fourcc_str, "on": True})
                break
        if not self._rec["on"]:
            raise RuntimeError(f"Failed to open VideoWriter for {path}. Install H.264 support or try .avi with XVID.")

        # Warm up the renderer once after previous resets
        _ = self.env.sim.render(camera_name=camera_name, height=H, width=W, depth=False)

        # Seed file with a few frames so it’s never near-empty
        for _ in range(3):
            self._video_frame("start")

    def _video_stop(self):
        if getattr(self, "_rec", None) and self._rec.get("on", False):
            self._rec["writer"].release()
            print(f"[VIDEO] Saved to {self._rec['path']} (codec={self._rec.get('fourcc')}, frames={self._rec['frames']})")
            self._rec["on"] = False

    def run(self):
        # run inference and save video
        self._video_start(f"Q5_O-trial_{trial_num}.mp4", fps=30, H=256, W=256, camera_name="birdview")
        print("[VIDEO] Recording trial {trial_num} to {video_path}...")


        self._video_frame()
        self._video_stop()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Save OpenVLA video data to a file.")
    parser.add_argument("input_path", type=str, help="Path to the input OpenVLA video data.")
    parser.add_argument("output_path", type=str, help="Path to save the output video file.")
    args = parser.parse_args()

    experiment = Experiment(args.input_path, args.output_path)
    experiment.run()
