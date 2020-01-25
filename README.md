# atari_rl
This repository includes trained dqn agent to play Breakout (Atari Game) using Keras/Tensorflow and Open Gym AI.

# This repository includes:
- **train_agent.py** - python script to training agent. Training can be paused and resume. Score during training and other parameters are saved to **log_training** directory. Trained models are saved to **trained_network** directory.
- **test_agent.py** - python script to test trained agent. Testing can be recorded and saved as *.mp4 file. Results of testing are saved to **log_test.txt** file.
- **training_graph.py** - python script to generate graph from training log files.

## Agent description
Agent was learned by 37 hours (about 9k games). Score during training is shown on the bottom:

<img src="https://github.com/m-milena/atari_rl/blob/master/log_training/training_score.png" width="400">

Progress during training:
<img src="https://github.com/m-milena/atari_rl/blob/master/agent_progress/after_5h/openaigymvideo017368video000000.gif" width="400">


