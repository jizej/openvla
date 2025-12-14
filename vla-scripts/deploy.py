"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path], predict_waypoint: int) -> str:
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    elif predict_waypoint == 0:
        return f"In: What action waypoint best estabilishes the pregrasp pose to perform the task: {instruction.lower()}?\nOut:"
    elif predict_waypoint == 1:
        return f"In: What action waypoint best estabilishes the grasp pose to perform the task: {instruction.lower()}?\nOut:"
    elif predict_waypoint == 2:
        return f"In: What action waypoint best estabilishes the release pose to perform the task: {instruction.lower()}?\nOut:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"



# === Server Interface ===
class OpenVLAServer:
    def __init__(self, openvla_path: Union[str, Path], stats_path: Union[str, Path], attn_implementation: Optional[str] = "flash_attention_2", cfg=None) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        if cfg is not None:
            if cfg.predict_waypoint == -1:
                self.openvla_path = cfg.openvla_path
                self.stats_path = cfg.stats_path
            elif cfg.predict_waypoint == 0:
                self.openvla_path = cfg.openvla_path_pregrasp
                self.stats_path = cfg.stats_path_pregrasp
            elif cfg.predict_waypoint == 1:
                self.openvla_path = cfg.openvla_path_grasp
                self.stats_path = cfg.stats_path_grasp
            elif cfg.predict_waypoint == 2:
                self.openvla_path = cfg.openvla_path_release
                self.stats_path = cfg.stats_path_release
        else:
            self.openvla_path = openvla_path 
            self.stats_path = stats_path
        self.attn_implementation = attn_implementation
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.predict_waypoint = cfg.predict_waypoint if cfg is not None else -1

        print(f"Using openvla path: {self.openvla_path}")

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if os.path.isdir(self.stats_path):
            with open(Path(self.stats_path) / "dataset_statistics.json", "r") as f:
                if self.predict_waypoint == -1:
                    self.vla.norm_stats = json.load(f)
                else:
                    norm_stats = json.load(f)
                    idx = self.predict_waypoint
                    norm_stats = {
                        "cereal": {
                            "action": {
                                "mean": norm_stats["cereal"]["action"]["mean"][idx],
                                "std": norm_stats["cereal"]["action"]["std"][idx],
                                "max": norm_stats["cereal"]["action"]["max"][idx],
                                "min": norm_stats["cereal"]["action"]["min"][idx],
                                "q01": norm_stats["cereal"]["action"]["q01"][idx],
                                "q99": norm_stats["cereal"]["action"]["q99"][idx],
                                "mask": norm_stats["cereal"]["action"]["mask"]
                            },
                            "proprio": {
                                "mean": norm_stats["cereal"]["proprio"]["mean"][idx],
                                "std": norm_stats["cereal"]["proprio"]["std"][idx],
                                "max": norm_stats["cereal"]["proprio"]["max"][idx],
                                "min": norm_stats["cereal"]["proprio"]["min"][idx],
                                "q01": norm_stats["cereal"]["proprio"]["q01"][idx],
                                "q99": norm_stats["cereal"]["proprio"]["q99"][idx],
                            },
                            "num_transitions": norm_stats["cereal"]["num_transitions"],
                            "num_trajectories": norm_stats["cereal"]["num_trajectories"]
                        }
                    }
                    self.vla.norm_stats = norm_stats

    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            image, instruction = payload["image"], payload["instruction"]
            unnorm_key = payload.get("unnorm_key", None)
            predict_mode = payload.get("predict_mode", "action")
            if predict_mode == "action":
                predict_waypoint = -1
            elif predict_mode == "pregrasp":
                predict_waypoint = 0
            elif predict_mode == "grasp":
                predict_waypoint = 1
            elif predict_mode == "release":
                predict_waypoint = 2
            else:
                print(f"Invalid predict mode: {predict_mode}")
                predict_waypoint = -1

            if os.path.isdir(self.stats_path):
                if predict_mode == "action":
                    stats_path = cfg.stats_path 
                elif predict_mode == "pregrasp":
                    stats_path = cfg.stats_path_pregrasp
                elif predict_mode == "grasp":
                    stats_path = cfg.stats_path_grasp
                elif predict_mode == "release":
                    stats_path = cfg.stats_path_release
                else:
                    stats_path = cfg.stats_path
                norm_stats = json.load(open(Path(stats_path) / "dataset_statistics.json", "r"))
                print(f"predict_mode: {predict_mode}")
                print(f"Using stats path: {stats_path}")
                if predict_mode == "action":
                    self.vla.norm_stats = norm_stats
                else:
                    mode2idx = {"action": -1, "pregrasp": 0, "grasp": 1, "release": 2}
                    idx = mode2idx[predict_mode]
                    norm_stats = {
                        "cereal": {
                            "action": {
                                "mean": norm_stats["cereal"]["action"]["mean"][idx],
                                "std": norm_stats["cereal"]["action"]["std"][idx],
                                "max": norm_stats["cereal"]["action"]["max"][idx],
                                "min": norm_stats["cereal"]["action"]["min"][idx],
                                "q01": norm_stats["cereal"]["action"]["q01"][idx],
                                "q99": norm_stats["cereal"]["action"]["q99"][idx],
                                "mask": norm_stats["cereal"]["action"]["mask"]
                            },
                            "proprio": {
                                "mean": norm_stats["cereal"]["proprio"]["mean"][idx],
                                "std": norm_stats["cereal"]["proprio"]["std"][idx],
                                "max": norm_stats["cereal"]["proprio"]["max"][idx],
                                "min": norm_stats["cereal"]["proprio"]["min"][idx],
                                "q01": norm_stats["cereal"]["proprio"]["q01"][idx],
                                "q99": norm_stats["cereal"]["proprio"]["q99"][idx],
                            },
                            "num_transitions": norm_stats["cereal"]["num_transitions"],
                            "num_trajectories": norm_stats["cereal"]["num_trajectories"]
                        }
                    }
                    self.vla.norm_stats = norm_stats
            # Run VLA Inference
            prompt = get_openvla_prompt(instruction, self.openvla_path, predict_waypoint)
            print(f"Prompt: {prompt}")
            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
            action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off
    # openvla_path: Union[str, Path] = "checkpoints/openvla-7b+my_cereal_task_trajectories+b20+lr-0.0005+lora-r32+dropout-0.0+251211_0203"
    # stats_path: Union[str, Path] = "checkpoints/openvla-7b+my_cereal_task_trajectories+b20+lr-0.0005+lora-r32+dropout-0.0+251211_0203"
    
    # openvla_path = "checkpoints/openvla-7b+my_cereal_task_trajectories+b20+lr-0.0005+lora-r32+dropout-0.0+251212_0455"
    # stats_path = "checkpoints/openvla-7b+my_cereal_task_trajectories+b20+lr-0.0005+lora-r32+dropout-0.0+251212_0455"

    openvla_path = "checkpoints/openvla-7b+cereal+b20+lr-0.0005+lora-r32+dropout-0.0+release+251213_1412"
    openvla_path = "checkpoints/openvla-7b+cereal+b20+lr-0.0005+lora-r32+dropout-0.0+pregrasp+251213_1434"
    openvla_path = "checkpoints/openvla-7b+cereal+b20+lr-0.0005+lora-r32+dropout-0.0+grasp+251213_1456"
    stats_path = openvla_path
    # openvla_path: Union[str, Path] = "openvla/openvla-7b"

    predict_waypoint: int = -1                                        # -1: predict action, 0: predict pregrasp, 1: predict grasp, 2: predict release
    openvla_path_trajectory = "checkpoints/openvla-7b+my_cereal_task_trajectories+b20+lr-0.0005+lora-r32+dropout-0.0+251212_0455"
    stats_path_trajectory = openvla_path_trajectory
    openvla_path_pregrasp = "checkpoints/openvla-7b+cereal+b20+lr-0.0005+lora-r32+dropout-0.0+pregrasp+251212_1505"
    stats_path_pregrasp = openvla_path_pregrasp
    openvla_path_grasp = "checkpoints/openvla-7b+cereal+b20+lr-0.0005+lora-r32+dropout-0.0+grasp+251212_1522"
    stats_path_grasp = openvla_path_grasp
    openvla_path_release = "checkpoints/openvla-7b+cereal+b20+lr-0.0005+lora-r32+dropout-0.0+release+251212_1547"
    stats_path_release = openvla_path_release

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port

    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAServer(cfg.openvla_path, cfg.stats_path, attn_implementation="flash_attention_2", cfg=cfg)
    # change port 
    # cfg.port += cfg.predict_waypoint
    server.run(cfg.host, cfg.port)


if __name__ == "__main__":
    import argparse
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_waypoint", type=int, default=-1)
    args = parser.parse_args()
    cfg = DeployConfig(predict_waypoint=args.predict_waypoint)
    deploy()
