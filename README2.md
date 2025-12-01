# Deep Q-Network for Atari Breakout

## Overview

This project implements and compares multiple Deep Q-Network (DQN) variants for playing Atari Breakout, achieving a best score of **843 points** with an average evaluation score of **460 points**.

![Best Run](best-videos/best.gif)

### Implemented Algorithms
- **Vanilla DQN** (Mnih et al., 2015)
- **Double DQN** (van Hasselt et al., 2016)
- **Dueling DQN** (Wang et al., 2016)
- **Distributional DQN (C51)** (Bellemare et al., 2017)
- **Noisy Networks** (Fortunato et al., 2017)

## Project Structure

```
.
├── agent/
│   ├── agent.py          # Agent implementation
│   ├── env.py            # Environment wrapper
│   ├── memory.py         # Replay buffer
│   └── model.py          # Neural network architectures
├── model/
│   └── [run_name]/
│       ├── checkpoints/  # Saved model weights
│       ├── config.yaml   # Run configuration
│       ├── logs/         # Training metrics
│       └── videos/       # Evaluation recordings
├── config.yaml           # Default hyperparameters
├── main.py              # Training script
├── test.py              # Evaluation script
└── visualize.py         # Plotting utilities
```

## Methodology

### Hyperparameter Decisions

**Episodic Life** (`episodic_life=True`)  
Sends end-of-episode signals when losing a life without resetting the environment. This approach teaches the agent that losing lives is detrimental while maintaining state continuity across lives, preventing the agent from only experiencing initial game states.

**Reward Clipping** (`clip_rewards=True`)  
While unclipped rewards might theoretically help the agent learn tunneling strategies, in practice they destabilize training by disrupting gradient flow and Q-value estimation. Clipped rewards produce faster, more stable convergence.

**Frame Skip** (`frame_skip=4`)  
Tested values: 2, 3, and 4. While frame skip of 4 occasionally causes the agent to miss the ball by small margins, lower values converge significantly slower without clear performance benefits. Frame skip of 4 was retained for training efficiency.

**Replay Memory** (`buffer_size=300k`)  
Limited by hardware constraints. Disk-based replay using TorchRL proved too slow for practical use.

**Action Space Randomization** (`noop_max=30`)  
Up to 30 no-op actions at episode start reduce determinism and improve generalization.

**Exploration Strategy**  
Epsilon-greedy exploration outperformed Noisy Networks, providing better control and reducing action looping behaviors.

## Experimental Results

All models were trained for approximately 7 hours per run due to hardware limitations.

### Vanilla DQN (Baseline)

The standard DQN implementation served as our baseline, achieving stable learning and reaching an average reward of 280 by 30k episodes.

### Distributional DQN (C51)

According to the Rainbow paper (Hessel et al., 2018), distributional learning provides significant gains on Breakout. However, our implementation faced stability challenges.

**Initial Run:**
![C51 Run 1](plots/metrics_plot_C51.png)

Training progressed well until episode 10k, where gradient explosion caused NaN values in Q-predictions.

**Stabilized Run:**
![C51 Run 2](plots/metrics_plot_C51_2.png)

After implementing gradient clipping and reward normalization, training stabilized but convergence became extremely slow (average reward of 10 at 50k episodes vs. 280 for vanilla DQN at 30k).

**Best Configuration:**
![C51 Run 3](plots/metrics_plot_C51_3.png)

Despite extensive hyperparameter tuning (atom counts, value ranges), C51 underperformed compared to simpler architectures. Given the complexity of the distributional Bellman equation and limited computational budget, we proceeded to evaluate simpler improvements.

### Double Dueling DQN (D3QN)

Based on Rainbow findings that Double and Dueling modifications provide consistent gains on Breakout, we combined both techniques.

**Initial Results:**
![D3QN Run 1](plots/metrics_plot_D3QN.png)

Promising early results with healthy Q-value growth and increasing average rewards.

**Optimized Configuration:**
![D3QN Run 2](plots/metrics_plot_D3QN2.png)

High average rewards were achieved, but evaluation videos revealed a critical issue: **action looping**.

#### The Looping Problem

![Looping Behavior](plots/looping.gif)

The agent became stuck in repetitive action patterns after clearing the first wall. Occasionally, it would reach the second wall but immediately lose all remaining lives:

![Second Wall Failure](plots/second_wall.gif)

**Root Cause Analysis:**
1. **Determinism:** Despite no-op randomization, only 30 unique initial states exist. As epsilon decays, the agent exploits learned patterns that lead to loops.
2. **Visual Score Dependency:** The agent observes the score pixels, creating different state representations after breaking through walls. This causes policy degradation in novel high-score states.

### Solutions Explored

#### 1. Fine-Tuning on End-Game States

**Approach:** Pre-fill replay buffer with high-score experiences (score > 360) and maintain a ratio of full games to end-game states during training.

![Fine-Tuning Results](plots/metrics_plot_FineTune.png)

**Outcome:** Catastrophic forgetting occurred despite attempts to balance replay ratios, maintain small epsilon values, and limit training episodes. Q-values and rewards both declined steadily.

![Fine-Tuning Instability](plots/metrics_plot2_FineTune.png)

#### 2. Penalty-Based Shaping

**Life Loss Penalty (-1 reward):** Incentivizes life preservation, potentially allowing the agent to survive longer after breaking through the second wall.

**Loop Detection Penalty (-1 reward):** Applied when no bricks are hit for 200 consecutive steps to discourage repetitive behaviors.

![Penalty Results](plots/metrics_plot2_D3QNPenalty.png)

**Outcome:** Best configuration using both penalties achieved an average evaluation score of 385 (10-game average), outperforming standard D3QN. The initial Q-value drop reflects early life losses before the agent learns survival strategies.

#### 3. Adaptive Exploration (Best Solution)

**Approach:** Dynamically adjust epsilon based on agent behavior:
- Increase epsilon by 0.005 every 100 steps without brick contact
- Reset epsilon to baseline when bricks are hit

This maintains exploitation of learned policies while injecting exploration when loop patterns emerge.

![Adaptive Exploration Results](plots/metrics_plot_D3QN4.png)

**Breakthrough Performance:**
- Average reward: **460**
- Best evaluation: **460** (consistent second-wall clearing)
- Best single game: **843 points**

The continued upward trend suggests further improvements with extended training.

## Final Results

**Best Model:** Double Dueling DQN with Adaptive Exploration  
**Highest Score:** 843 points  
**Average Evaluation Score:** 460 points  

## Future Work

1. **Extended Training:** Performance curves indicate room for improvement with additional compute time.
2. **C51 Optimization:** Better hyperparameter search might unlock distributional learning benefits.
3. **Alternative Algorithms:** PPO-based approaches have achieved 804 average reward ([reference](https://huggingface.co/cleanrl/Breakout-v5-cleanba_ppo_envpool_impala_atari_wrapper-seed2)).
4. **Hardware Upgrades:** Larger replay buffers and longer training runs would benefit all approaches.

## References

- Mnih et al. (2015). Human-level control through deep reinforcement learning.
- van Hasselt et al. (2016). Deep Reinforcement Learning with Double Q-learning.
- Wang et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning.
- Bellemare et al. (2017). A Distributional Perspective on Reinforcement Learning.
- Hessel et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning.
- Fortunato et al. (2017). Noisy Networks for Exploration.

## Installation & Usage

```bash
# Install dependencies
poetry install

# Train model
python main.py

# Evaluate trained model (you need to change the path of the model that will be loaded inside the script)
python replay.py

# Visualize metrics
python visualize.py
```

See `config.yaml` for all available hyperparameters.