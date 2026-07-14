import os
import csv
from time import time

import numpy as np
from tqdm import trange
import vizdoom as vzd

from env.doom_env import preprocess, create_game
from config.settings import RESOLUTION
from training.episode_recorder import record_episode_from_actions


def run_training_a2c(
    game,
    agent,
    actions,
    num_epochs,
    frame_repeat,
    steps_per_epoch,
    csv_path,
    discount_factor,
    rollout_len,
    shaping_cfg,
    video_dir=None,
    label="A2C",
):
    start_time = time()
    episode_rewards = []
    episode_kills = []
    episode_shots = []
    epoch_mean_rewards = []

    # ⭐ NEW — path for epoch-level logging
    epoch_csv_path = csv_path.replace(".csv", "_epoch.csv")

    buttons = game.get_available_buttons()
    attack_idx = buttons.index(vzd.Button.ATTACK) if vzd.Button.ATTACK in buttons else None

    # =========================================================================================
    # Episode-level CSV (unchanged)
    # =========================================================================================
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "env_total_reward",
            "kills",
            "shots",
            "accuracy",
            "mean_loss",
            "mean_value",
            "elapsed_sec",
        ])

    # =========================================================================================
    # ⭐ NEW: Epoch-level CSV
    # =========================================================================================
    with open(epoch_csv_path, "w", newline="") as ef:
        epoch_writer = csv.writer(ef)
        epoch_writer.writerow([
            "epoch",
            "mean_reward",
            "mean_kills",
            "mean_shots",
            "mean_accuracy",
            "mean_loss",
            "mean_value",
            "elapsed_sec",
        ])

    # =========================================================================================
    # TRAINING LOOP
    # =========================================================================================
    for epoch in range(num_epochs):
        print(f"\n[{label}] Epoch #{epoch + 1}")

        game.new_episode()
        epoch_scores = []
        kills = 0
        shots = 0
        steps_in_epoch = 0

        epoch_episode_data = []
        current_actions = []

        # Stats to compute epoch aggregates
        epoch_kills_list = []
        epoch_shots_list = []
        epoch_loss_list = []
        epoch_value_list = []

        # =========================================================================================
        # ROLLOUT LOOP
        # =========================================================================================
        while steps_in_epoch < steps_per_epoch:
            rewards = []
            dones = []
            values = []
            log_probs = []
            entropies = []
            last_next_state = None
            last_done = False

            for _ in trange(rollout_len, leave=False):
                if game.is_episode_finished():
                    ep_r = game.get_total_reward()
                    episode_rewards.append(ep_r)
                    episode_kills.append(kills)
                    episode_shots.append(shots)
                    epoch_scores.append(ep_r)

                    acc = kills / shots if shots > 0 else 0.0
                    mean_loss, mean_val = agent.get_and_reset_stats()
                    elapsed = time() - start_time

                    # Record episode-level row
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            len(episode_rewards),
                            ep_r,
                            kills,
                            shots,
                            acc,
                            mean_loss,
                            mean_val,
                            elapsed,
                        ])

                    epoch_kills_list.append(kills)
                    epoch_shots_list.append(shots)
                    epoch_loss_list.append(mean_loss)
                    epoch_value_list.append(mean_val)

                    epoch_episode_data.append((ep_r, list(current_actions)))
                    game.new_episode()

                    kills = 0
                    shots = 0
                    current_actions = []

                if steps_in_epoch >= steps_per_epoch:
                    break

                if game.is_episode_finished():
                    game.new_episode()

                state = preprocess(game.get_state().screen_buffer, RESOLUTION)
                a_idx, value, log_prob, entropy = agent.act(state)
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

                rewards.append(shaped_reward)
                dones.append(done)
                values.append(value)
                log_probs.append(log_prob)
                entropies.append(entropy)

                last_next_state = next_state
                last_done = done

                steps_in_epoch += 1
                if steps_in_epoch >= steps_per_epoch:
                    break

            if len(rewards) > 0:
                agent.update_rollout(
                    rewards,
                    dones,
                    values,
                    log_probs,
                    entropies,
                    last_next_state,
                    last_done,
                )

        # =========================================================================================
        # EPOCH SUMMARY & CSV LOGGING
        # =========================================================================================
        if len(epoch_scores) == 0:
            mean_reward = 0.0
        else:
            mean_reward = np.mean(epoch_scores)

        epoch_mean_rewards.append(mean_reward)

        mean_kills = np.mean(epoch_kills_list) if epoch_kills_list else 0.0
        mean_shots = np.mean(epoch_shots_list) if epoch_shots_list else 0.0
        mean_accuracy = mean_kills / mean_shots if mean_shots > 0 else 0.0
        mean_loss = np.mean(epoch_loss_list) if epoch_loss_list else 0.0
        mean_value = np.mean(epoch_value_list) if epoch_value_list else 0.0
        elapsed = time() - start_time

        print(
            f"[{label}] Epoch {epoch+1}: "
            f"mean={mean_reward:.2f}, "
            f"elapsed={(elapsed)/60.0:.2f} min"
        )

        # ⭐ NEW — write epoch-level summary
        with open(epoch_csv_path, "a", newline="") as ef:
            epoch_writer = csv.writer(ef)
            epoch_writer.writerow([
                epoch + 1,
                mean_reward,
                mean_kills,
                mean_shots,
                mean_accuracy,
                mean_loss,
                mean_value,
                elapsed,
            ])

        # =========================================================================================
        # SAVE BEST EPISODE VIDEO FOR THIS EPOCH
        # =========================================================================================
        if video_dir is not None and len(epoch_episode_data) > 0:
            best_ep_r, best_actions = max(epoch_episode_data, key=lambda x: x[0])
            out_path = os.path.join(video_dir, f"epoch_{epoch+1:03d}_best.mp4")
            record_episode_from_actions(
                create_game, actions, best_actions, frame_repeat, out_path, fps=8
            )

    game.close()
    return agent, episode_rewards, episode_kills, episode_shots, epoch_mean_rewards
