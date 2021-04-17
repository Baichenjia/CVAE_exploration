# Conditional VAE for Exploration (Baseline of VDM)


## Prerequisites

VDM requires python3.6,
tensorflow-gpu 1.13.1 or 1.14.0,
tensorflow-probability 0.6.0,
openAI [baselines](https://github.com/openai/baselines),
openAI [Gym](http://gym.openai.com/),
openAI [Retro](https://github.com/openai/retro)

## Usage

### Atari games

The following command should train a pure exploration agent on "Breakout" with default experiment parameters.

```
python run.py --env BreakoutNoFrameskip-v4
```

### Atari games with sticky actions

The following command should train a pure exploration agent on "sticky Breakout" with a probability of 0.25

```
python run.py --env BreakoutNoFrameskip-v4 --stickyAtari
```

