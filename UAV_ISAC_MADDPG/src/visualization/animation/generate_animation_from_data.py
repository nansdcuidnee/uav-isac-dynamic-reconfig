import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

# Add project root so `src.*` imports work when running this file directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.visualization.animation.animation_2d import (  # noqa: E402
    create_comparison_animation,
    create_trajectory_animation,
)
from src.visualization.animation.animation_3d import (  # noqa: E402
    create_3d_comparison_animation,
    create_3d_trajectory_animation,
)


def _load_user_positions(user_json_path):
    with open(user_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Current trainer output: list[{'user_id','x','y','z'}]
    if isinstance(data, list):
        user_pos = [[float(item["x"]), float(item["y"]), float(item.get("z", 0.0))] for item in data]
        return user_pos

    # Legacy format fallback: {'user_positions': [[x,y,z], ...]}
    if isinstance(data, dict) and "user_positions" in data:
        return [[float(p[0]), float(p[1]), float(p[2] if len(p) > 2 else 0.0)] for p in data["user_positions"]]

    raise ValueError(f"Unsupported user JSON format: {user_json_path}")


def _select_episode(episodes, episode):
    # episodes: iterable[int]
    episodes = sorted(set(episodes))
    if not episodes:
        raise ValueError("No episode data found in trajectory JSON")

    if episode == "last":
        return episodes[-1]
    if episode == "first" or episode == "default":
        return episodes[0]

    # Allow explicit episode index, e.g. --episode 3
    try:
        ep = int(episode)
    except ValueError as exc:
        raise ValueError(f"Invalid --episode value: {episode}") from exc

    if ep not in episodes:
        raise ValueError(f"Episode {ep} not found. Available episodes: {episodes}")
    return ep


def _build_connections(user_pos, drone_pos):
    connections = []
    for user in user_pos:
        if not drone_pos:
            connections.append(None)
            continue
        dists = [(user[0] - d[0]) ** 2 + (user[1] - d[1]) ** 2 for d in drone_pos]
        closest = int(np.argmin(dists))
        connections.append(closest)
    return connections


def _frames_from_flat_list(trajectory_data, user_pos, episode_selector):
    # Expected trainer output: list of dicts
    # [{'episode':0,'step':0,'uav_id':0,'x':...,'y':...,'z':...}, ...]
    selected_episode = _select_episode((item["episode"] for item in trajectory_data), episode_selector)
    items = [item for item in trajectory_data if int(item["episode"]) == selected_episode]

    by_step = defaultdict(list)
    for item in items:
        by_step[int(item["step"])].append(item)

    uav_ids = sorted({int(item["uav_id"]) for item in items})
    history = {uid: [] for uid in uav_ids}
    frames = []

    for step in sorted(by_step.keys()):
        entries = sorted(by_step[step], key=lambda x: int(x["uav_id"]))
        pos_map = {
            int(e["uav_id"]): [float(e["x"]), float(e["y"]), float(e.get("z", 0.0))]
            for e in entries
        }
        drone_pos = [pos_map[uid] for uid in uav_ids if uid in pos_map]

        for uid in uav_ids:
            if uid in pos_map:
                history[uid].append(pos_map[uid])

        frames.append(
            {
                "drone_pos": drone_pos,
                "user_pos": user_pos,
                "connections": _build_connections(user_pos, drone_pos),
                "trajectories": [history[uid].copy() for uid in uav_ids],
                "step": step,
            }
        )

    return frames


def _frames_from_legacy_dict(trajectory_data, user_pos, episode_selector):
    # Legacy fallback format:
    # {"episode_0": [{"drone_positions": [[x,y,z], ...]}, ...], ...}
    episode_keys = list(trajectory_data.keys())
    if not episode_keys:
        raise ValueError("Empty trajectory JSON")

    # Sort keys by trailing number when possible
    def key_rank(k):
        parts = str(k).split("_")
        try:
            return int(parts[-1])
        except ValueError:
            return 0

    sorted_keys = sorted(episode_keys, key=key_rank)

    if episode_selector == "last":
        episode_key = sorted_keys[-1]
    elif episode_selector in ("first", "default"):
        episode_key = sorted_keys[0]
    else:
        # explicit int selector
        idx = int(episode_selector)
        matched = [k for k in sorted_keys if key_rank(k) == idx]
        if not matched:
            raise ValueError(f"Episode {idx} not found in legacy trajectory JSON")
        episode_key = matched[0]

    episode_data = trajectory_data[episode_key]
    if not episode_data:
        return []

    num_drones = len(episode_data[0].get("drone_positions", []))
    history = {i: [] for i in range(num_drones)}
    frames = []

    for step, item in enumerate(episode_data):
        drone_pos = [[float(p[0]), float(p[1]), float(p[2] if len(p) > 2 else 0.0)] for p in item.get("drone_positions", [])]
        for i, pos in enumerate(drone_pos):
            history[i].append(pos)

        frames.append(
            {
                "drone_pos": drone_pos,
                "user_pos": user_pos,
                "connections": _build_connections(user_pos, drone_pos),
                "trajectories": [history[i].copy() for i in range(num_drones)],
                "step": step,
            }
        )

    return frames


def load_trajectory_from_json(trajectory_json_path, user_json_path, episode="last"):
    with open(trajectory_json_path, "r", encoding="utf-8") as f:
        trajectory_data = json.load(f)

    user_pos = _load_user_positions(user_json_path)

    if isinstance(trajectory_data, list):
        return _frames_from_flat_list(trajectory_data, user_pos, episode)
    if isinstance(trajectory_data, dict):
        return _frames_from_legacy_dict(trajectory_data, user_pos, episode)

    raise ValueError(f"Unsupported trajectory JSON format: {trajectory_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate animation from trajectory JSON data")
    parser.add_argument("--trajectory_json", type=str, required=True, help="Path to trajectory_data.json")
    parser.add_argument("--user_json", type=str, required=True, help="Path to user_positions.json")
    parser.add_argument("--episode",
        type=str,
        default="last",
        help="Episode selector: 'last', 'first' (or 'default'), or an explicit integer episode id",
    )
    parser.add_argument("--output", type=str, default="results/videos/demo_3d.mp4", help="Output video path")
    parser.add_argument("--mode", type=str, default="3d", choices=["2d", "3d", "comparison"], help="Animation mode")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--downsample", type=str, default="auto", help="Downsample rate (auto or integer)")

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("Loading trajectory JSON...")
    frames_data = load_trajectory_from_json(args.trajectory_json, args.user_json, args.episode)
    print(f"Loaded {len(frames_data)} frames")

    # 处理 downsample 参数
    if args.downsample == "auto":
        downsample = "auto"
    else:
        try:
            downsample = int(args.downsample)
        except ValueError:
            print(f"Invalid downsample value: {args.downsample}, using 'auto' instead")
            downsample = "auto"

    if args.mode == "2d":
        create_trajectory_animation(frames_data, save_path=args.output, fps=args.fps, downsample=downsample)
    elif args.mode == "3d":
        create_3d_trajectory_animation(frames_data, save_path=args.output, fps=args.fps, downsample=downsample)
    else:
        # Placeholder comparison: compare same data on both sides when only one run is provided.
        create_3d_comparison_animation(frames_data, frames_data, save_path=args.output, fps=args.fps, downsample=downsample)


if __name__ == "__main__":
    main()
