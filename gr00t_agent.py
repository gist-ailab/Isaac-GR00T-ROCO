# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np
from PIL import Image

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Galaxea_Lab_External.tasks  # noqa: F401

from gr00t.policy.server_client import PolicyClient


def apply_vlm_processing(images: np.ndarray, language: str, processor):
        """
        Args:
            batch:
                video: [T, C, H, W]
        Returns: vlm_content format for collation
        """
        # Convert images to PIL format
        pil_images = [Image.fromarray(v.numpy()) for v in images]

        # Create conversation with images and text
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": language},
                    *[{"type": "image", "image": img} for img in pil_images],
                ],
            }
        ]

        # Apply chat template but don't process yet - let collator handle it
        text = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        # Return vlm_content format for collation
        return {
            "vlm_content": {
                "text": text,
                "images": pil_images,
                "conversation": conversation,
            }
        }

def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # exit()
    policy = PolicyClient(host='127.0.0.1', port='5555')
    if not policy.ping(): # Verify connection
        raise RuntimeError("Cannot connect to policy server!")

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    obs, info = env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():            
            # obs = env.env._get_observations()
            # sample actions from -1 to 1
            # actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # action = torch.zeros_like(actions)
            # apply actions
            obs = obs['policy']
            obs_ = dict()

            for k, v in obs.items():
                 print(k, v.shape)
            state = torch.cat([obs['left_arm_joint_pos'], 
                               obs['left_gripper_joint_pos'][None, :], 
                               obs['left_gripper_joint_pos'][None, :], 
                               obs['right_arm_joint_pos'], 
                               obs['right_gripper_joint_pos'][None, :],
                               obs['right_gripper_joint_pos'][None, :]],
                               dim = 1
                               )
            obs_['video'] = {'observations.images.head': obs['head_rgb'].unsqueeze(0).cpu().numpy(), #observations.images.head
                             'observations.images.left_hand': obs['left_hand_rgb'].unsqueeze(0).cpu().numpy(),
                             'observations.images.right_hand': obs['right_hand_rgb'].unsqueeze(0).cpu().numpy()
                             }
            obs_['state'] = {'state': np.array(state.unsqueeze(0).cpu(), dtype=np.float32)}
            obs_['language'] = {'annotation.human.action.task_description': [['gearbox assembly demos updated']]}
            action, info = policy.get_action(obs_)
            action = torch.Tensor(action['action'])

            # First action and Remove duplicated gripper joint pos
            print('action', action.shape, info)
            for k, v in obs_['state'].items():
                print(k, v, v.shape)

            for i in range(1):
                action_ = action[0, i]
                # action_ = torch.cat([action_[:6], action_[8:-2], action_[6:7], action_[15:16]], dim=0)[None, :]
                print(action_.shape)
                action_ = action_[[0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 6, 14]]

                print(action_)

                obs, reward, terminated, truncated, info = env.step(action_)
            # ["input_ids", "attention_mask", "pixel_values"]
            
            # print(obs.keys())
            # print(obs['policy'].keys())
            # print(obs['policy']['rgb'].shape)
            # print(obs['policy']['left_hand_rgb'].shape)
            # print(obs['policy']['right_hand_rgb'].shape)
            # print(obs['policy']['left_arm_joint_pos'].shape)
            # print(obs['policy']['right_arm_joint_pos'].shape)
            # print(obs['policy']['left_gripper_joint_pos'].shape)
            # print(obs['policy']['right_gripper_joint_pos'].shape)
            # env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
