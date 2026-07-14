import argparse
import os

import numpy as np
import torch

from config.paths import (
    BASELINE_DIR,
    NOISY_DIR,
    A2C_DIR,
    COMP_DIR,
)
from config.settings import (
    LEARNING_RATE,
    DISCOUNT_FACTOR,
    TRAIN_EPOCHS,
    LEARNING_STEPS_PER_EPOCH,
    REPLAY_MEMORY_SIZE,
    BATCH_SIZE,
    FRAME_REPEAT,
    A2C_ROLLOUT_LEN,
    A2C_GAE_LAMBDA,
    A2C_VALUE_LOSS_COEF,
    A2C_ENTROPY_COEF,
    make_shaping_cfg,
)
from env.doom_env import create_game, build_actions, preprocess
from agents.dqn_agent import DQNAgent
from agents.a2c_agent import A2CAgent
from training.train_dqn import run_training_dqn
from training.train_a2c import run_training_a2c
from metrics.plots import plot_series_comparison
from metrics.comparison_table import (
    write_epoch_reward_table,
    write_comparison_table,
)
from metrics.moving_statistics import moving_mean_and_var
from config.paths import ensure_dir

import csv
import os
from time import time

class EpochLogger:
    def __init__(self, log_path, fieldnames):
        self.log_path = log_path
        self.fieldnames = fieldnames

        # Write header once
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

    def log(self, data: dict):
        """Append a single row of epoch-level info."""
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(data)


def test_agent(game, agent, actions, n_episodes, frame_repeat, label="", return_episodes=False):
    results = []
    best_reward = -1e9
    best_actions = None

    buttons = game.get_available_buttons()
    attack_idx = buttons.index(vzd.Button.ATTACK) if vzd.Button.ATTACK in buttons else None

    for ep in range(n_episodes):
        game.new_episode()
        episode_actions = []

        kills = 0
        shots = 0
        total_reward = 0
        steps = 0

        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer, RESOLUTION)

            # Universal: DQN / Noisy / A2C all expose .get_action()
            a_idx = agent.get_action(state)
            episode_actions.append(a_idx)

            if attack_idx is not None and actions[a_idx][attack_idx] == 1:
                shots += 1

            r = game.make_action(actions[a_idx], frame_repeat)
            steps += 1

            if r > 0:
                kills += int(round(r))

            total_reward += r

        accuracy = kills / shots if shots > 0 else 0.0

        results.append({
            "reward": total_reward,
            "kills": kills,
            "shots": shots,
            "accuracy": accuracy,
            "steps": steps,
        })

        print(
            f"[Eval {label}] Episode {ep+1}/{n_episodes} | "
            f"R={total_reward:.2f}, K={kills}, S={shots}, Acc={accuracy:.2f}, Steps={steps}"
        )

        # Track best episode for video export
        if total_reward > best_reward:
            best_reward = total_reward
            best_actions = episode_actions

    if return_episodes:
        return results, best_actions
    else:
        return np.array([r["reward"] for r in results])



def train_all_agents(args):
    shaping_cfg = make_shaping_cfg()
    if args.disable_shaping:
        shaping_cfg["accuracy_bonus"] = 0.0
        shaping_cfg["shot_penalty"] = 0.0
    if args.disable_nstep:
        shaping_cfg["n_step"] = 1

    tmp_game = create_game(window_visible=False)
    actions, attack_idx = build_actions(tmp_game)
    n_actions = len(actions)
    print("Available buttons:", tmp_game.get_available_buttons())
    print("Number of discrete actions:", n_actions)
    tmp_game.close()

    baseline_rewards = []
    noisy_rewards = []
    a2c_rewards = []
    baseline_kills = []
    noisy_kills = []
    a2c_kills = []
    baseline_shots = []
    noisy_shots = []
    a2c_shots = []
    baseline_epoch_means = []
    noisy_epoch_means = []
    a2c_epoch_means = []

    if args.agent in ("all", "baseline"):
        ensure_dir(BASELINE_DIR)
        baseline_csv = os.path.join(BASELINE_DIR, "baseline_metrics.csv")
        game = create_game(window_visible=False)
        agent_baseline = DQNAgent(
            n_actions=n_actions,
            memory_size=REPLAY_MEMORY_SIZE,
            batch_size=BATCH_SIZE,
            discount=DISCOUNT_FACTOR,
            lr=LEARNING_RATE,
            model_path=os.path.join(BASELINE_DIR, "baseline_model.pth"),
            noisy=False,
            load_model=args.resume,
            target_update_interval=args.target_update_interval or 1000,
        )
        (
            agent_baseline,
            baseline_rewards,
            baseline_kills,
            baseline_shots,
            baseline_epoch_means,
        ) = run_training_dqn(
            game,
            agent_baseline,
            actions,
            num_epochs=TRAIN_EPOCHS,
            frame_repeat=FRAME_REPEAT,
            steps_per_epoch=LEARNING_STEPS_PER_EPOCH,
            csv_path=baseline_csv,
            discount_factor=DISCOUNT_FACTOR,
            shaping_cfg=shaping_cfg,
            video_dir=BASELINE_DIR,
            label="Baseline_DQN",
        )
        np.save(os.path.join(BASELINE_DIR, "baseline_rewards.npy"), baseline_rewards)
        np.save(os.path.join(BASELINE_DIR, "baseline_kills.npy"), baseline_kills)
        np.save(os.path.join(BASELINE_DIR, "baseline_shots.npy"), baseline_shots)
        torch.save(
            agent_baseline.q_net.state_dict(),
            os.path.join(BASELINE_DIR, "baseline_model.pth"),
        )

    if args.agent in ("all", "noisy"):
        ensure_dir(NOISY_DIR)
        noisy_csv = os.path.join(NOISY_DIR, "noisy_metrics.csv")
        game = create_game(window_visible=False)
        agent_noisy = DQNAgent(
            n_actions=n_actions,
            memory_size=REPLAY_MEMORY_SIZE,
            batch_size=BATCH_SIZE,
            discount=DISCOUNT_FACTOR,
            lr=LEARNING_RATE,
            model_path=os.path.join(NOISY_DIR, "noisy_model.pth"),
            noisy=not args.no_noisy,
            load_model=args.resume,
            target_update_interval=args.target_update_interval or 1000,
        )
        (
            agent_noisy,
            noisy_rewards,
            noisy_kills,
            noisy_shots,
            noisy_epoch_means,
        ) = run_training_dqn(
            game,
            agent_noisy,
            actions,
            num_epochs=TRAIN_EPOCHS,
            frame_repeat=FRAME_REPEAT,
            steps_per_epoch=LEARNING_STEPS_PER_EPOCH,
            csv_path=noisy_csv,
            discount_factor=DISCOUNT_FACTOR,
            shaping_cfg=shaping_cfg,
            video_dir=NOISY_DIR,
            label="Noisy_DQN",
        )
        np.save(os.path.join(NOISY_DIR, "noisy_rewards.npy"), noisy_rewards)
        np.save(os.path.join(NOISY_DIR, "noisy_kills.npy"), noisy_kills)
        np.save(os.path.join(NOISY_DIR, "noisy_shots.npy"), noisy_shots)
        torch.save(
            agent_noisy.q_net.state_dict(),
            os.path.join(NOISY_DIR, "noisy_model.pth"),
        )

    if args.agent in ("all", "a2c"):
        ensure_dir(A2C_DIR)
        a2c_csv = os.path.join(A2C_DIR, "a2c_metrics.csv")
        game = create_game(window_visible=False)
        agent_a2c = A2CAgent(
            n_actions=n_actions,
            discount=DISCOUNT_FACTOR,
            lr=LEARNING_RATE,
            value_loss_coef=A2C_VALUE_LOSS_COEF,
            entropy_coef=A2C_ENTROPY_COEF,
            gae_lambda=A2C_GAE_LAMBDA,
        )
        (
            agent_a2c,
            a2c_rewards,
            a2c_kills,
            a2c_shots,
            a2c_epoch_means,
        ) = run_training_a2c(
            game,
            agent_a2c,
            actions,
            num_epochs=TRAIN_EPOCHS,
            frame_repeat=FRAME_REPEAT,
            steps_per_epoch=LEARNING_STEPS_PER_EPOCH,
            csv_path=a2c_csv,
            discount_factor=DISCOUNT_FACTOR,
            rollout_len=A2C_ROLLOUT_LEN,
            shaping_cfg=shaping_cfg,
            video_dir=A2C_DIR,
            label="A2C",
        )
        np.save(os.path.join(A2C_DIR, "a2c_rewards.npy"), a2c_rewards)
        np.save(os.path.join(A2C_DIR, "a2c_kills.npy"), a2c_kills)
        np.save(os.path.join(A2C_DIR, "a2c_shots.npy"), a2c_shots)
        torch.save(
            agent_a2c.net.state_dict(),
            os.path.join(A2C_DIR, "a2c_model.pth"),
        )

    if args.agent == "baseline":
        noisy_rewards = noisy_kills = noisy_shots = noisy_epoch_means = []
        a2c_rewards = a2c_kills = a2c_shots = a2c_epoch_means = []
    elif args.agent == "noisy":
        baseline_rewards = baseline_kills = baseline_shots = baseline_epoch_means = []
        a2c_rewards = a2c_kills = a2c_shots = a2c_epoch_means = []
    elif args.agent == "a2c":
        baseline_rewards = baseline_kills = baseline_shots = baseline_epoch_means = []
        noisy_rewards = noisy_kills = noisy_shots = noisy_epoch_means = []

    ensure_dir(COMP_DIR)
    if len(baseline_rewards) and len(noisy_rewards) and len(a2c_rewards):
        plot_series_comparison(
            baseline_rewards,
            noisy_rewards,
            a2c_rewards,
            ylabel="Reward",
            title="Reward (moving average)",
            filename="learning_curves_rewards.png",
            window=20,
        )
        plot_series_comparison(
            baseline_kills,
            noisy_kills,
            a2c_kills,
            ylabel="Kills",
            title="Kills (moving average)",
            filename="learning_curves_kills.png",
            window=20,
        )
        baseline_acc = [
            (k / s) if s > 0 else 0.0 for k, s in zip(baseline_kills, baseline_shots)
        ]
        noisy_acc = [
            (k / s) if s > 0 else 0.0 for k, s in zip(noisy_kills, noisy_shots)
        ]
        a2c_acc = [
            (k / s) if s > 0 else 0.0 for k, s in zip(a2c_kills, a2c_shots)
        ]
        plot_series_comparison(
            baseline_acc,
            noisy_acc,
            a2c_acc,
            ylabel="Accuracy",
            title="Accuracy (moving average)",
            filename="learning_curves_accuracy.png",
            window=20,
        )

    if args.agent == "all":
        write_epoch_reward_table(
            baseline_epoch_means,
            noisy_epoch_means,
            a2c_epoch_means,
        )
        comparison_csv = os.path.join(COMP_DIR, "agent_comparison_table.csv")
        agents_data = {
            "Baseline DQN": {
                "rewards": baseline_rewards,
                "kills": baseline_kills,
                "shots": baseline_shots,
            },
            "Noisy DQN": {
                "rewards": noisy_rewards,
                "kills": noisy_kills,
                "shots": noisy_shots,
            },
            "A2C": {
                "rewards": a2c_rewards,
                "kills": a2c_kills,
                "shots": a2c_shots,
            },
        }
        write_comparison_table(comparison_csv, agents_data)


def eval_agent(args):
    # Create temporary game to determine action space
    tmp_game = create_game(window_visible=False)
    actions, _ = build_actions(tmp_game)
    n_actions = len(actions)
    tmp_game.close()

    # ------------------------------------------------------------------
    # Load Agent
    # ------------------------------------------------------------------
    if args.agent == "baseline":
        model_path = os.path.join(BASELINE_DIR, "baseline_model.pth")
        agent = DQNAgent(
            n_actions=n_actions,
            memory_size=1,
            batch_size=1,
            discount=DISCOUNT_FACTOR,
            lr=LEARNING_RATE,
            model_path=model_path,
            noisy=False,
            load_model=True,
        )

    elif args.agent == "noisy":
        model_path = os.path.join(NOISY_DIR, "noisy_model.pth")
        agent = DQNAgent(
            n_actions=n_actions,
            memory_size=1,
            batch_size=1,
            discount=DISCOUNT_FACTOR,
            lr=LEARNING_RATE,
            noisy=True,
            model_path=model_path,
            load_model=True,
        )

    elif args.agent == "a2c":
        model_path = os.path.join(A2C_DIR, "a2c_model.pth")
        agent = A2CAgent(
            n_actions=n_actions,
            discount=DISCOUNT_FACTOR,
            lr=LEARNING_RATE,
            value_loss_coef=A2C_VALUE_LOSS_COEF,
            entropy_coef=A2C_ENTROPY_COEF,
            gae_lambda=A2C_GAE_LAMBDA,
        )
        state_dict = torch.load(model_path, map_location="cpu")
        agent.net.load_state_dict(state_dict)
        agent.net.eval()  # Important!

    else:
        raise ValueError("Unknown agent for eval")

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    game = create_game(window_visible=args.record_video)

    results, best_episode_actions = test_agent(
        game=game,
        agent=agent,
        actions=actions,
        n_episodes=args.eval_episodes,
        frame_repeat=FRAME_REPEAT,
        label=args.agent,
        return_episodes=True,        # ⭐ NEW
    )

    game.close()

    rewards = np.array([r["reward"] for r in results])
    print(f"\nEval completed for {args.agent}. mean reward = {rewards.mean():.2f}")

    # ------------------------------------------------------------------
    # Optional: Save eval results CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(EVAL_DIR, f"{args.agent}_eval.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "kills", "shots", "accuracy", "steps"])
        for i, r in enumerate(results, 1):
            writer.writerow([
                i,
                r["reward"],
                r["kills"],
                r["shots"],
                r["accuracy"],
                r["steps"],
            ])

    print(f"Saved eval results to {csv_path}")

    # ------------------------------------------------------------------
    # Optional: save best episode video
    # ------------------------------------------------------------------
    if args.record_video and best_episode_actions is not None:
        out_path = os.path.join(EVAL_DIR, f"{args.agent}_best_eval.mp4")
        record_episode_from_actions(
            create_game,
            actions,
            best_episode_actions,
            FRAME_REPEAT,
            out_path,
            fps=8,
        )
        print(f"Recorded best episode video to {out_path}")



def parse_args():
    parser = argparse.ArgumentParser(description="ViZDoom Defend-the-Line RL Suite")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train agents")
    train_p.add_argument(
        "--agent",
        type=str,
        default="all",
        choices=["all", "baseline", "noisy", "a2c"],
        help="Which agent to train",
    )
    train_p.add_argument(
        "--disable-shaping",
        action="store_true",
        help="Disable accuracy-based reward shaping",
    )
    train_p.add_argument(
        "--disable-nstep",
        action="store_true",
        help="Disable n-step returns (use 1-step TD)",
    )
    train_p.add_argument(
        "--no-noisy",
        action="store_true",
        help="Use plain DQN instead of NoisyNet for 'noisy' agent",
    )
    train_p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved model weights if present",
    )
    train_p.add_argument(
        "--target-update-interval",
        type=int,
        default=None,
        help="Override target network update interval for DQN",
    )

    eval_p = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_p.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["baseline", "noisy", "a2c"],
        help="Which agent to evaluate",
    )
    eval_p.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate",
    )
    eval_p.add_argument(
        "--record-video",
        action="store_true",
        help="Show window while evaluating",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "train":
        train_all_agents(args)
    elif args.command == "eval":
        eval_agent(args)
