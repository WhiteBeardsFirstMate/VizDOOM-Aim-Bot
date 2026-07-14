import os
import csv
import numpy as np

from config.paths import COMP_DIR


def write_epoch_reward_table(
    baseline_epoch_means,
    noisy_epoch_means,
    a2c_epoch_means,
):
    out_path = os.path.join(COMP_DIR, "epoch_reward_summary.csv")
    max_epochs = max(
        len(baseline_epoch_means),
        len(noisy_epoch_means),
        len(a2c_epoch_means),
    )
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "baseline_mean_reward",
            "noisy_mean_reward",
            "a2c_mean_reward",
        ])
        for e in range(max_epochs):
            writer.writerow([
                e + 1,
                baseline_epoch_means[e] if e < len(baseline_epoch_means) else "",
                noisy_epoch_means[e] if e < len(noisy_epoch_means) else "",
                a2c_epoch_means[e] if e < len(a2c_epoch_means) else "",
            ])
    print(f"Saved epoch reward summary → {out_path}")


def write_comparison_table(out_path, agents_data):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Agent",
            "Episodes",
            "Avg Reward",
            "Avg Kills",
            "Avg Shots",
            "Avg Accuracy",
            "Best Reward",
            "Best Accuracy",
        ])

        for agent_name, d in agents_data.items():
            rewards = np.array(d["rewards"], dtype=np.float32)
            kills = np.array(d["kills"], dtype=np.float32)
            shots = np.array(d["shots"], dtype=np.float32)
            if len(rewards) == 0:
                continue
            accuracy = kills / np.maximum(shots, 1.0)

            writer.writerow([
                agent_name,
                len(rewards),
                rewards.mean(),
                kills.mean(),
                shots.mean(),
                accuracy.mean(),
                rewards.max(),
                accuracy.max(),
            ])

    print(f"\nSaved agent comparison table → {out_path}")

    print("\n=== Performance Comparison Table ===")
    for agent_name, d in agents_data.items():
        rewards = np.array(d["rewards"], dtype=np.float32)
        kills = np.array(d["kills"], dtype=np.float32)
        shots = np.array(d["shots"], dtype=np.float32)
        if len(rewards) == 0:
            continue
        accuracy = kills / np.maximum(shots, 1.0)
        print(f"\n[{agent_name}]")
        print(f" Episodes:   {len(rewards)}")
        print(f" Avg Reward: {rewards.mean():.2f}")
        print(f" Avg Kills:  {kills.mean():.2f}")
        print(f" Avg Shots:  {shots.mean():.2f}")
        print(f" Avg Acc:    {accuracy.mean():.3f}")
        print(f" Best Reward:{rewards.max():.2f}")
        print(f" Best Acc:   {accuracy.max():.3f}")
