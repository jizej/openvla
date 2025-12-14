import robosuite as suite
from robosuite.utils import camera_utils
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import cv2
# --- 1. IMPORT camera_utils ---
from robosuite.utils import camera_utils

# --- Import your new environment to register it ---
try:
    import robosuite.environments.manipulation.pick_place_clutter
except ImportError:
    print("=" * 80)
    print("ERROR: Could not import PickPlaceClutter environment.")
    print("Please make sure 'pick_place_clutter.py' is in 'robosuite/environments/manipulation/'")
    print("and you have added 'from . import pick_place_clutter' to that folder's __init__.py")
    print("=" * 80)
    exit()

import os
import sys
import time
import json
import pprint
import signal
import datetime
import threading
import requests
import json_numpy
json_numpy.patch()

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

from merge import merge_lora

# from generate_vla_dataset_trajectories import VLADataGenerator
from generate_vla_dataset_visualize import VLADataGenerator
from visualize_dataset_and_affordances import visualize_dataset

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VideoEnv():
    """
    A robosuite env with video saving
    """
    def __init__(self, env, obs, cfg):
        self.env = env
        self.cfg = cfg

        self.obs = obs
        self.robot_pos = self.obs["robot0_eef_pos"].copy()
        self.robot_quat = self._get_current_quat()
        self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)
        self._rec = {} # For video saving

        # 5. Camera parameters
        self.cam_width = self.env.camera_widths[0]
        self.cam_height = self.env.camera_heights[0]

    def _get_current_quat(self):
#        """Helper to consistently get the correct quaternion from observations."""
#        if "robot0_eef_quat_site" in self.obs:
#            return self.obs["robot0_eef_quat_site"].copy()
#        else:
#            return self.obs["robot0_eef_quat"].copy()

        return self.obs["robot0_eef_quat_site"].copy()  # SciPy-friendly [x,y,z,w]

    def video_frame(self, text=None):
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


        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bgr = np.ascontiguousarray(bgr)

        # Track frames-written to catch “empty” files
        self._rec["frames"] += 1
        self._rec["writer"].write(bgr)
        
    def video_start(self, path="eval.mp4", fps=30, H=256, W=256, camera_name="agentview"):
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
            self.video_frame("start")

    def video_stop(self):
        if getattr(self, "_rec", None) and self._rec.get("on", False):
            self._rec["writer"].release()
            print(f"[VIDEO] Saved to {self._rec['path']} (codec={self._rec.get('fourcc')}, frames={self._rec['frames']})")
            self._rec["on"] = False
    
    def step(self, action):
        """
        step the environment and record a video frame
        """
        # Ensure action is a writable numpy array before passing to robosuite
        action_np = np.array(action, dtype=np.float64, copy=True)
        if not action_np.flags.writeable:
            action_np = action_np.copy()

        self.obs, _, _, _ = self.env.step(action_np)
        self.video_frame()


@dataclass
class EvalConfig:
    # fmt: off
    exp_id: str = None                                              # Unique experiment ID (will be initialized if left None)
    exp_tag: str = None                                             # Extra tag to end onto the end of experiment ID string

    # Directory Paths
    output_dir: Path = Path("eval_output")                               # Path to directory to store model output


@draccus.wrap()
def eval(cfg: EvalConfig) -> None:
    task_generator = VLADataGenerator("eval_task")

    # task_generator.run_generation_loop(num_trials=1,
    #                                     num_videos=1,
    #                                     start_index=0)
    task_generator.instruction = task_generator.generate_instruction("cereal box", "target bin")

    # compute gt waypoints and save to json
    # --- CAPTURE "HOME" POSE (A4) ---
    # This is the pose we'll return to.
    neutral_pos = task_generator.robot_pos.copy()
    neutral_quat = task_generator.robot_quat.copy()

    # Get Camera Intrinsics and Extrinsics, ONLY FOR VISUALIZATION
    cam_name = "agentview"
    K = camera_utils.get_camera_intrinsic_matrix(
        task_generator.env.sim, cam_name, task_generator.cam_height, task_generator.cam_width
    )
    T_wc = camera_utils.get_camera_extrinsic_matrix(
        task_generator.env.sim, cam_name
    )
    
    # --- IMAGE CAPTURE ---
    # 1. Get initial images (AT THE START)
    rgb_image_raw = task_generator.obs["agentview_image"]
    depth_image_raw = task_generator.obs["agentview_depth"]
    
    # 2. Process RGB
    # Flip vertically (MuJoCo origin is bottom-left)
    rgb_image = cv2.flip(rgb_image_raw, 0)
    
    # 3. Process Depth
    # Convert to meters
    depth_real = camera_utils.get_real_depth_map(task_generator.env.sim, depth_image_raw)
    # Flip vertically to match RGB
    depth_image = cv2.flip(depth_real, 0)
    
    # 4. Get 3D poses and instruction
    cereal_body_id = task_generator.env.obj_body_id["Cereal"]
    cereal_pos = task_generator.env.sim.data.body_xpos[cereal_body_id]
    cereal_mj_quat = task_generator.env.sim.data.body_xquat[cereal_body_id]

    target_bin_pos = task_generator.env.target_bin_placements[2]
    
    # 5. Calculate 7-DOF Waypoint Labels
    print("  Analyzing object orientation ...")
    
    # 1. Convert MuJoCo quat [w, x, y, z] to SciPy quat [x, y, z, w]
    cereal_scipy_quat = np.array([
        cereal_mj_quat[1], cereal_mj_quat[2], cereal_mj_quat[3], cereal_mj_quat[0]
    ])
    
    # 2. Get the cereal box's rotation from its quaternion
    cereal_rotation = R.from_quat(cereal_scipy_quat)
    
    # 3. Get the "long side" (local X-axis) as a world-frame vector
    #    This is the first column of the rotation matrix.
    box_long_side_vector = cereal_rotation.as_matrix()[:, 0]
    print(f"  Detected box long side vector (world): {np.round(box_long_side_vector, 2)}")

    # 4. Calculate the perpendicular grasp vector (90-deg rotation in XY plane)
    #    This vector is perpendicular to the *long side*, so it's aligned with the *short side*.
    #    (dx, dy) -> (-dy, dx)
    x_axis_gripper = np.array([-box_long_side_vector[1], box_long_side_vector[0], 0])

    # 5. Normalize the vector
    norm = np.linalg.norm(x_axis_gripper)
    if norm < 1e-5:
        # happens if the box's long side is pointing straight up/down
        print("  Warning: Box long side is Z-aligned. Defaulting to world X-axis grasp.")
        x_axis_gripper = np.array([1., 0., 0.])
    else:
        x_axis_gripper /= norm
    print(f"  Calculated grasp X-axis (world): {np.round(x_axis_gripper, 2)}")

    # 6. Define Z-axis as "straight down"
    z_axis_gripper = np.array([0., 0., -1.])
    
    # 7. Find Y-axis via cross product
    y_axis_gripper = np.cross(z_axis_gripper, x_axis_gripper)
    
    # 8. Build rotation matrix and get quaternion
    grasp_rotation_matrix = np.array([x_axis_gripper, y_axis_gripper, z_axis_gripper]).T
    grasp_rot = R.from_matrix(grasp_rotation_matrix)
    grasp_quat = grasp_rot.as_quat()

    # 9. Define the standard "placing" orientation
    standard_place_rot = R.from_euler('xyz', [180, 0, 90], degrees=True)
    standard_place_quat = standard_place_rot.as_quat()
    
    # A1: Pre-Grasp
    a1_pos = cereal_pos + np.array([0, 0, 0.30])
    # A2: Grasp
    a2_pos = cereal_pos + np.array([0, 0, 0.03])
    # A3: Release
    a3_pos = target_bin_pos + np.array([0, 0, 0.10])
    a4_pos = neutral_pos # Use the "home" pos we just saved
    
    # Use helper to create 7-DOF poses
    waypoint_labels = {
        "A1_pregrasp": task_generator._get_7dof_pose(a1_pos, grasp_quat, -1.0), # Gripper Open
        "A2_grasp":    task_generator._get_7dof_pose(a2_pos, grasp_quat,  1.0), # Gripper Closed
        "A3_release":  task_generator._get_7dof_pose(a3_pos, standard_place_quat, -1.0), # Gripper Open
        "A4_home":     task_generator._get_7dof_pose(a4_pos, neutral_quat, -1.0)  # Gripper Open
    }
    
    # trajectory_goals_3d = {
    #         "A1_pregrasp": (a1_pos.tolist(), grasp_quat.tolist()),
    #         "A2_grasp": (a2_pos.tolist(), grasp_quat.tolist()),
    #         "A3_release": (a3_pos.tolist(), standard_place_quat.tolist()),
    # }
    
    # 7. Save data IF successful
    task_generator.save_sample(
        trial_idx=0,
        rgb_img=rgb_image,
        depth_img=depth_image,
        instruction=task_generator.instruction,
        waypoint_labels=waypoint_labels,
        K=K,           # <-- ADD THIS only for visualization
        T_wc=T_wc      # <-- ADD THIS only for visualization
    )


    # save env to video env and eval model
    task_env = task_generator.env
    video_env = VideoEnv(task_env, task_generator.obs, cfg)
    video_env.video_start()
    task_generator._video_start()
    neutral_quat = task_generator.robot_quat.copy()

    print(f"Instruction: {task_generator.instruction}")

    img = video_env.obs["agentview_image"]
    img = cv2.flip(img, 0).astype(np.uint8)
    # save the image
    cv2.imwrite("eval_image_waypoint.png", img)

    waypoint_1_action = requests.post(
        "http://0.0.0.0:8000/act",
        json={"image": img.copy(), "instruction": task_generator.instruction, "predict_mode": "pregrasp"}
    ).json()
    print(f"Waypoint 1 Action: {waypoint_1_action}")

    waypoint_2_action = requests.post(
        "http://0.0.0.0:8000/act",
        json={"image": img.copy(), "instruction": task_generator.instruction, "predict_mode": "grasp"}
    ).json()
    print(f"Waypoint 2 Action: {waypoint_2_action}")

    waypoint_3_action = requests.post(
        "http://0.0.0.0:8000/act",
        json={"image": img.copy(), "instruction": task_generator.instruction, "predict_mode": "release"}
    ).json()
    print(f"Waypoint 3 Action: {waypoint_3_action}")
    
    a1_pos = np.array(waypoint_1_action[:3], dtype=np.float64, copy=True)
    a2_pos = np.array(waypoint_2_action[:3], dtype=np.float64, copy=True)
    a3_pos = np.array(waypoint_3_action[:3], dtype=np.float64, copy=True)

    # Convert predicted rotvec -> quaternion (robosuite expects quat for slerp)
    pregrasp_quat = R.from_euler('xyz', np.array(waypoint_1_action[3:6], dtype=np.float64, copy=True)).as_quat()
    grasp_quat = R.from_euler('xyz', np.array(waypoint_2_action[3:6], dtype=np.float64, copy=True)).as_quat()
    standard_place_quat = R.from_euler('xyz', np.array(waypoint_3_action[3:6], dtype=np.float64, copy=True)).as_quat()

    trajectory_goals_3d = {
        "A1_pregrasp": (a1_pos.tolist(), pregrasp_quat.tolist()),
        "A2_grasp": (a2_pos.tolist(), grasp_quat.tolist()),
        "A3_release": (a3_pos.tolist(), standard_place_quat.tolist()),
    }
    waypoint_labels_predicted = {
        "A1_pregrasp": task_generator._get_7dof_pose(a1_pos, pregrasp_quat, -1.0), # Gripper Open
        "A2_grasp":    task_generator._get_7dof_pose(a2_pos, grasp_quat,  1.0), # Gripper Closed
        "A3_release":  task_generator._get_7dof_pose(a3_pos, standard_place_quat, -1.0), # Gripper Open
        "A4_home":     task_generator._get_7dof_pose(a4_pos, neutral_quat, -1.0)  # Gripper Open
    }
    task_generator.save_sample(
        trial_idx=1,
        rgb_img=rgb_image,
        depth_img=depth_image,
        instruction=task_generator.instruction,
        waypoint_labels=waypoint_labels_predicted,
        K=K,           # <-- ADD THIS only for visualization
        T_wc=T_wc      # <-- ADD THIS only for visualization
    )
    visualize_dataset(task_generator.output_dir, "eval_output_visualize", num_samples=2)
    # task_generator.move_to_pose(a1_pos, neutral_quat, -1.0, count=60)
    # task_generator.move_to_pose(a2_pos, neutral_quat, -1.0, count=40)
    # task_generator.move_to_pose(a3_pos, neutral_quat, -1.0, count=70)
    is_success = task_generator.execute_trajectory(trajectory_goals_3d)

    if is_success:
        print("Trajectory successful")
    else:
        print("Trajectory failed")

    video_env.video_stop()
    task_generator._video_stop()

if __name__ == "__main__":
    eval()