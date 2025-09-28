import os
import json
import shutil
from PIL import Image
import numpy as np

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import tyro

REPO_NAME = "your_hf_username/rlds_dataset_3"  # customize this


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset (adjust shapes to match your data)
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="custom_robot",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),  # ee_state length = 7
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),  # same dimension as state
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop through all episodes
    for episode_name in sorted(os.listdir(data_dir)):
        episode_path = os.path.join(data_dir, episode_name)
        if not os.path.isdir(episode_path):
            continue

        json_path = os.path.join(episode_path, f"{episode_name}.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        img_dir = os.path.join(episode_path, "img")
        ee_states = data["ee_states"]
        img_files = data["img_filenames"]
        task_instruction = "put white cube on tray"

        # Iterate through timesteps (state_t, image_t → action = state_{t+1})
        for t in range(len(ee_states) - 1):
            img_path = os.path.join(img_dir, img_files[t])
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)

            state = np.array(ee_states[t], dtype=np.float32)
            action = np.array(ee_states[t + 1], dtype=np.float32)

            dataset.add_frame(
                {
                    "image": img,
                    "state": state,
                    "actions": action,
                    "task": task_instruction
                }
            )

        dataset.save_episode()

    # Optionally push to Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["rlds", "robotics", "custom"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
