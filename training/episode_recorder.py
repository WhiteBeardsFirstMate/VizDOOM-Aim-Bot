import os
import imageio.v2 as imageio
import numpy as np
from config.paths import ensure_dir

def record_episode_from_actions(create_game_fn, actions, action_indices,
                                frame_repeat, out_path, fps=8):
    game = create_game_fn(window_visible=False)
    ensure_dir(os.path.dirname(out_path))

    # ---- FIX: enforce mp4 output so imageio uses ffmpeg ----
    if not out_path.endswith(".mp4"):
        out_path = out_path + ".mp4"

    frames = []

    game.new_episode()
    for a_idx in action_indices:
        if game.is_episode_finished():
            break
        game.make_action(actions[a_idx], frame_repeat)
        if not game.is_episode_finished():
            frame = game.get_state().screen_buffer
            frame_rgb = np.stack([frame] * 3, axis=-1).astype(np.uint8)
            frames.append(frame_rgb)

    with imageio.get_writer(out_path, fps=fps) as writer:
        for f in frames:
            writer.append_data(f)

    game.close()
    print(f"Saved episode video ? {out_path}")
