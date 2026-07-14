# ViZDoom Reinforcement Learning Suite

DQN • Noisy-DQN • A2C • Reward Shaping • N-Step Returns • Automatic Video Logging • Full Metrics Export

Structure:

project/
agents/
    dqn_agent.py
    a2c_agent.py
    __init__.py

config/
    paths.py
    settings.py
    __init__.py

env/
    doom_env.py
    __init__.py

metrics/
    plots.py
    comparison_table.py
    moving_statistics.py
    __init__.py

networks/
    a2c_network.py
    dueling_dqn.py
    noisy_dueling_dqn.py
    __init__.py

training/
    episode_recorder.py
    train_dqn.py
    train_a2c.py
    utils.py
    __init__.py
main.py
__init__.py

--------------------------------------------------------------------------------------------------------

Required Packages:
    
    torch==2.2.0
    torchvision
    torchaudio
    vizdoom
    numpy
    scipy
    opencv-python
    matplotlib
    tqdm
    imageio
    scikit-image
    gym

--------------------------------------------------------------------------------------------------------

How To Use:

baseline → Standard DQN

noisy → NoisyNet DQN

a2c → Advantage Actor–Critic

all → Train all three sequentially (training only)

    python main.py train --agent <agent_name>

    python main.py eval --agent <agent_name>

To disable reward shaping

    python main.py train --agent all --disable-shaping

To disable n-step returns

    python main.py train --agent baseline --disable-nstep

To override target network update interval

    python main.py train --agent baseline --target-update-interval 500

To resume model training

    python main.py train --agent baseline --resume

--------------------------------------------------------------------------------------------------------

config/settings.py supplies:

    - epochs

    - learning rate

    - replay buffer size

    - frame repeat

    - discount factor

    - rollout length (A2C)

    - shaping parameters

    - number of steps per epoch


