# How to use this repository
1. install dependencies `pip install -r requirements.txt`
2. run pretrained agetnts `python -m scripts.run --load_model pip_pendulum --agents PIP --env SparsePendulum-v0`
3. train a new agent ` python -m scripts.train --name test_cacla --agent CACLA --rbf 5 5 17 --episodes 10000`. See the code to all available options.

## Acknowledgment
`scripts/train.py` and `replay_buffer.py` are adapted from https://github.com/sfujim/TD3


## Reference
C. Wulur, C. Weber and S. Wermter, "Planning-integrated Policy for Efficient Reinforcement Learning in Sparse-reward Environments", in International Joint Conference on Neural Network, 2021.