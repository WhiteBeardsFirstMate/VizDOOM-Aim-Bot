LEARNING_RATE = 0.00025
DISCOUNT_FACTOR = 0.99

TRAIN_EPOCHS = 10
LEARNING_STEPS_PER_EPOCH = 3000

BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 10000

FRAME_REPEAT = 4
RESOLUTION = (30, 45)

N_STEP_RETURNS = 3
TARGET_UPDATE_INTERVAL = 1000
MAX_GRAD_NORM = 5.0

A2C_ROLLOUT_LEN = 10
A2C_GAE_LAMBDA = 0.95
A2C_VALUE_LOSS_COEF = 0.5
A2C_ENTROPY_COEF = 0.01

ACCURACY_BONUS_SCALE = 0.2
SHOT_PENALTY_SCALE = 0.01

def make_shaping_cfg(accuracy_bonus=None, shot_penalty=None,
                     n_step=None, gamma=None):
    from config import settings as s
    return {
        "accuracy_bonus": s.ACCURACY_BONUS_SCALE if accuracy_bonus is None else accuracy_bonus,
        "shot_penalty": s.SHOT_PENALTY_SCALE if shot_penalty is None else shot_penalty,
        "n_step": s.N_STEP_RETURNS if n_step is None else n_step,
        "gamma": s.DISCOUNT_FACTOR if gamma is None else gamma,
    }
