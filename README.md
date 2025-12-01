# Atari breakout

Best run: 843 points
![Best](best-videos/best.gif)

Intro: this repo implements some DQN paper to play atari breakout
- DQN
- Double
- Duelling
- Distributional
- Noisy Linear (with torchrl)

Structure:
.
├── README.md
├── agent
│   ├── agent.py
│   ├── env.py
│   ├── memory.py
│   └── model.py
├── best-videos
├── config.yaml
├── convert.sh
├── copymain.py
├── main.py
├── model
│   ├── run name
│   │   ├── checkpoints
│   │   ├── config.yaml
│   │   ├── logs
│   │   └── videos
├── poetry.lock
├── pyproject.toml
├── replay.py
├── test.py
├── utils.py
└── visualize.py

Experiments:

A lot of experiments were made on the vanilla DQN without recording to find the right starting parameters. They can be found in `config.yaml` but here a quick explanation on the choice for the most important one:

Episodic life: send a end of episode signal to the agent without reseting the env. Allow the agent to understand that loosing a life is bad and the non env reset allows newt states (because if terminate env on life lose, agent will be stuck with first game step).

Clipped reward: we can think that if keeping unclipped reward can help the agent tunnel because it will learn that shooting on the same spot will make it gain more points, but actually it will completly disrupt the gradient and the estimation and we see that it converges way slower and the training is more unstable.

Frame skip = 4: when watching training and evaluation videos we see that with frame skip = 4, the speed is really high and the agent miss the ball but only with a small error. So maybe frame skip is too high so we try with 2 and 3. The agent converge way slower so we have not been capable of clearly evaluating. But with reference to other training it seems to perform less. So we keep 4 to have a fast training.

We have a replay memory of 300k because my laptop has limited RAM. We tried to write to disk and then read dynamically thanks to torchrl but it was very slow.

To be a bit less deterministic we set noop to 30. It means that up to 30 no ops actions will be done before starting the game.

We tried noisy linear (thanks to torchrl noisy linear) and epsilon greedy and we found that epsilon greedy gaves us more control and reduce the looping problem.


With these parameters set we tested multiple models. Each model was run during a whole night so approximatly 7 hours.

The first was the vanilla DQN from deepmind paper.


Reading the rainbow article they say that for breakout:
- PER and multi step decrease performance a lot
- Noisy linear it's mitigate
- Double and duelling slightly increase
- Distributional is a huge gain

We started implemeting C51. The big issue is the stability, here the result of the first run:

![C51](plots/metrics_plot_C51.png)

The reward goes well and then near 10K episode, we got NaN for Q and prediction so likely the gradient has exploded. So we clipped the gradient and the reward and here the second run:

![C51_2](plots/metrics_plot_C51_2.png)

There is no gradient exploding but the converge is really slow, at 50k episodes the average reward is still 10 likewise for the vanilla dqn it was already at 280 at 30k episodes. Reading articles they say it's better when rewards are not clipped. I tried multiple configurations, number of atoms and atoms range and here is the best run among all my experiments:

![C51_3](plots/metrics_plot_C51_3.png)

The reward is a little better than the previous version but still really low. Maybe it's not the optimal parameters and with more time i would have found the optimal combination but since i accorded a week on C51 and each runs = a whole night i decided to move on. Also it's possible that my implementation of C51 isn't completly accurate since the article is quite complex.

Bringing back the rainbow paper we tried to implement the simplest change so the double DQN and the duelling DQN. Here the result of the first run:

![D3QN](plots/metrics_plot_D3QN.png)

The results are promising, the average reward and the Q growth well so maybe with some parameters tweak we can obtain a good run. Here the best run among the parameter tunning:

![D3QN2](plots/metrics_plot_D3QN2.png)

The average reward is really high. But looking at some videos, we saw that the agent is stuck at the end of the first wall. It starts looping as you can see on the videos:

![looping](plots/looping.gif)

And sometimes with a very low probability is reachs the second wall but die instantly because he has no more lives :

![second_wall](plots/second_wall.gif)

Reading articles and blog posts some says that the looping problem came from the fact that breakout is deterministic so even with no op there is still only 30 different games and when the epsilon decay became really low the agent will be looping. As for why the agent die directly after the second wall, some says it's because since we take the whole image of the game, the agent can see the score, so it's not exactly the same as the start game since there is some pixels that are different. We tried training longer and changing parameters but we still faced this wall issue. After some times we tested different approach:
- fine tunning a model already good only on end parties.
- Adding penalties when loosing a life. So reward -1. In a hope to when the agent pass the second wall to have more life and so can go further. And a looping penatlies. So when the agent does not touch bricks after X step we gave him -1 has a reward.
- Increase slowy the epsilon until it breaks a brick.

1. Fine tunning on end parties

The idea is to have a good agent and let it play to fill the replay memory with only end game so when the score is greater than 360 (it's complelty arbitrary). We then train like normal and keep a ratio of full game and only end game. The results were catastrophic the agent forget everything. We tried having more full game than end game, having a small epsilon at the starts, running only on a small number of episodes and so on and the best run we had was this one:

![FineTune](plots/metrics_plot_FineTune.png)

As we can see the Q is dropping and the reward is dropping and stabilizing but slowly decreassing (it's an average over 100 episodes but when looking at each episodes the reward was decreassing and sometimes there is only one or two games near 400 and then ppor reward like below 10). As you can see on the following graphics reward are completly unstable:

![FineTune2](plots/metrics_plot2_FineTune.png)

2. Penalties

We tried adding two penalties, the first when is when the agent lose a life. Since it dies instantly after the second wall because he has no more life, maybe with more lives he can go further so to do so we need to teach him that lives are important.

The second is avoiding him to loop so when looping (we detect it by looking if it has not break a brick after 200 steps). We send -1 reward. We tried combination of the two so with only lives, only looping penalties and both. Both shows the best results:

![Penalty](plots/metrics_plot2_D3QNPenalty.png)

It reachs a better average reward than the D3QN basic and the best evaluation (evaluation is made on 10 complete games) is 385 average reward. We can see on the average Q that the Q drops below 0 at the beginning since it losses a lot of life and then go up because he starts learning to survive.

3. Control the exploration

The idea is to keep the right set of parameters i have that already know how to reach 300-350 reward as average but just increase the exploration when looping. Everytime the agent does not touch a break after 100 steps we increase the epsilon by 0.005 and when he touch a bricks we reset the epsilon at the value it was before. The results were shocking:

![D3QN4](plots/metrics_plot_D3QN4.png)

The average reward hit more than 400 and the best evaluation was 460 so it means he achieves to pass the second wall in average. When running the best model with the same strategy of exploration we achieves to get a game with 843 points. Maybe with a better computer i would have trained it longer since it seems it can go up further.

In conclusion:

We achieved a score of 843 points with an average of 460 with a Double Duelling DQN (D3QN) model on breakout with a modified epislon decay. We maybe can achiueve better results with C51 with the correct parameters, train longer our model or switch to another paradigm like PPO. When looking through hugging face i saw people reaching 804 average reward with PPO like this one: https://huggingface.co/cleanrl/Breakout-v5-cleanba_ppo_envpool_impala_atari_wrapper-seed2


