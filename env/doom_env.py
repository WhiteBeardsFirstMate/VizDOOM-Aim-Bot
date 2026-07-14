import itertools as it
import numpy as np
import skimage.transform
import vizdoom as vzd

from config.paths import CONFIG_FILE_PATH

def preprocess(img, resolution=(30, 45)):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def create_game(window_visible: bool = False) -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(CONFIG_FILE_PATH)
    game.set_window_visible(window_visible)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    return game

def build_actions(game: vzd.DoomGame):
    n_buttons = game.get_available_buttons_size()
    buttons = game.get_available_buttons()
    actions = [list(a) for a in it.product([0, 1], repeat=n_buttons)]
    attack_idx = buttons.index(vzd.Button.ATTACK) if vzd.Button.ATTACK in buttons else None
    return actions, attack_idx
