import os
import csv
from collections import deque
from time import time

import numpy as np
from tqdm import trange
import vizdoom as vzd

from env.doom_env import preprocess, create_game
from training.utils import flush_nstep
from config.settings import RESOLUTION
from training.episode_recorder import record_episode_from_actions


def run_training_dqn(
    game,
    agent,
    actions,
    num_epochs,
    frame_repeat,
    steps_per_epoch,
    csv_path,
    discount_factor,
    shaping_cfg,
    video_dir=None,
    label="DQN",
):
    start_time = time()
    episode_rewards = []
    episode_kills = []
    episode_shots = []
    epoch_mean_rewards = []

    # ⭐ NEW — epoch-level metrics CSV
    epoch_csv_path = csv_path.replace(".csv", "_epoch.csv")
    with open(epoch_csv_path, "w", newline="") as ef:
        epoch_writer = csv.writer(ef)
        epoch_writer.writerow([
            "epoch",
            "mean_reward",
            "mean_kills",
            "mean_shots",
            "mean_accuracy",
            "mean_loss",
            "mean_q",
            "elapsed_sec",
        ])

    buttons = game.get_available_buttons()
    attack_idx = buttons.index(vzd.Button.ATTACK) if vzd.Button.ATTACK in buttons else None

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "env_total_reward",
            "kills",
            "shots",
            "accuracy",
            "mean_loss",
            "mean_q",
            "elapsed_sec",
        ])

        # =========================================================================================
        # TRAINING LOOP
        # =========================================================================================
        for epoch in range(num_epochs):
            print(f"\n[{label}] Epoch #{epoch + 1}")
            game.new_episode()

            kills = 0
            shots = 0
            epoch_scores = []
            nstep_buffer = deque(maxlen=shaping_cfg["n_step"])
            global_step = 0

            epoch_episode_data = []
            current_actions = []

            # ⭐ NEW — Track per-epoch aggregates
            epoch_kills_list = []
            epoch_shots_list = []
            epoch_loss_list = []
            epoch_q_list = []

            while global_step < steps_per_epoch:
                if game.is_episode_finished():
                    # flush n-step transitions on episode end
                    while nstep_buffer:
                        R, s0, a0, ns, d, k = flush_nstep(nstep_buffer, discount_factor)
                        agent.append_memory(s0, a0, R, ns, d, k)

                    ep_r = game.get_total_reward()
                    episode_rewards.append(ep_r)
                    episode_kills.append(kills)
                    episode_shots.append(shots)
                    epoch_scores.append(ep_r)

                    acc = kills / shots if shots > 0 else 0.0
                    mean_loss, mean_q = agent.get_and_reset_stats()
                    elapsed = time() - start_time

                    writer.writerow([
                        len(episode_rewards),
                        ep_r,
                        kills,
                        shots,
                        acc,
                        mean_loss,
                        mean_q,
                        elapsed,
                    ])
                    f.flush()

                    # ⭐ NEW — append epoch-level stat trackers
                    epoch_kills_list.append(kills)
                    epoch_shots_list.append(shots)
                    epoch_loss_list.append(mean_loss)
                    epoch_q_list.append(mean_q)

                    epoch_episode_data.append((ep_r, list(current_actions)))

                    # reset episode
                    game.new_episode()
                    kills = 0
                    shots = 0
                    nstep_buffer.clear()
                    current_actions = []

                    if global_step >= steps_per_epoch:
                        break

                # Normal DQN step
                state = preprocess(game.get_state().screen_buffer, RESOLUTION)
                a_idx = agent.get_action(state)
                current_actions.append(a_idx)

                if attack_idx is not None and actions[a_idx][attack_idx] == 1:
                    shots += 1

                raw_r = game.make_action(actions[a_idx], frame_repeat)

                if raw_r > 0:
                    kills += int(round(raw_r))

                acc = kills / shots if shots > 0 else 0.0
                shaped_reward = (
                    raw_r
                    + shaping_cfg["accuracy_bonus"] * acc
                    - shaping_cfg["shot_penalty"] * shots
                )

                done = game.is_episode_finished()
                if not done:
                    next_state = preprocess(game.get_state().screen_buffer, RESOLUTION)
                else:
                    next_state = np.zeros((1, *RESOLUTION), dtype=np.float32)

                # N-step buffer push
                nstep_buffer.append((state, a_idx, shaped_reward, next_state, done))
                if len(nstep_buffer) == shaping_cfg["n_step"] or done:
                    R, s0, a0, ns, d, k = flush_nstep(nstep_buffer, discount_factor)
                    agent.append_memory(s0, a0, R, ns, d, k)

                # Train if enough memory
                if global_step > agent.batch_size:
                    agent.train_step()

                global_step += 1

            # =====================================================================================
            # ⭐ NEW — EPOCH SUMMARY
            # =====================================================================================
            if len(epoch_scores) == 0:
                mean_reward = 0.0
            else:
                mean_reward = np.mean(epoch_scores)

            epoch_mean_rewards.append(mean_reward)

            mean_kills = np.mean(epoch_kills_list) if epoch_kills_list else 0.0
            mean_shots = np.mean(epoch_shots_list) if epoch_shots_list else 0.0
            mean_accuracy = mean_kills / mean_shots if mean_shots > 0 else 0.0
            mean_loss = np.mean(epoch_loss_list) if epoch_loss_list else 0.0
            mean_q = np.mean(epoch_q_list) if epoch_q_list else 0.0
            elapsed = time() - start_time

            print(
                f"[{label}] Epoch {epoch+1}: "
                f"mean={mean_reward:.2f}, "
                f"elapsed={(elapsed)/60.0:.2f} min"
            )

            # ⭐ NEW — write epoch row
            with open(epoch_csv_path, "a", newline="") as ef:
                epoch_writer = csv.writer(ef)
                epoch_writer.writerow([
                    epoch + 1,
                    mean_reward,
                    mean_kills,
                    mean_shots,
                    mean_accuracy,
                    mean_loss,
                    mean_q,
                    elapsed,
                ])

            # =====================================================================================
            # Best-episode video export (unchanged)
            # =====================================================================================
            if video_dir is not None and len(epoch_episode_data) > 0:
                best_ep_r, best_actions = max(epoch_episode_data, key=lambda x: x[0])
                out_path = os.path.join(video_dir, f"epoch_{epoch+1:03d}_best.mp4")
                record_episode_from_actions(
                    create_game, actions, best_actions, frame_repeat, out_path, fps=8
                )

    game.close()
    return agent, episode_rewards, episode_kills, episode_shots, epoch_mean_rewards
