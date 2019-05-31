# Improving Exploration in SAC with Normalizing Flows Policies

This codebase was used to generate the results documented in the paper "[Improving Exploration in Soft-Actor-Critic with Normalizing Flows Policies](arxiv_url_placeholder)".
Patrick Nadeem Ward<sup>*12</sup>, Ariella Smofsky<sup>*12</sup>, Avishek Joey Bose<sup>12</sup>. INNF Workshop ICML 2019.

* <sup>*</sup> Equal contribution, <sup>1</sup> McGill University,  <sup>2</sup> Mila
* Correspondence to:
  * Patrick Nadeem Ward <[Github: NadeemWard](https://github.com/NadeemWard), patrick.ward@mail.mcgill.ca>
  * Ariella Smofsky <[Github: asmoog](https://github.com/asmoog), ariella.smofsky@mail.mcgill.ca>

## Requirements
* [PyTorch](https://pytorch.org/)
* [comet.ml](https://www.comet.ml/)

## Run Experiments
Gaussian policy on Dense Gridworld environment with REINFORCE:
```
TODO
```

Gaussian policy on Sparse Gridworld environment with REINFORCE:
```
TODO
```

Gaussian policy on Dense Gridworld environment with reparametrization:
```
python main.py --namestr=G-S-DG-CG --make_cont_grid --batch_size=128 --replay_size=100000 --hidden_size=64 --num_steps=100000 --policy=Gaussian --smol --comet --dense_goals --silent
```

Gaussian policy on Sparse Gridworld environment with reparametrization:
```
python main.py --namestr=G-S-CG --make_cont_grid --batch_size=128 --replay_size=100000 --hidden_size=64 --num_steps=100000 --policy=Gaussian --smol --comet --silent
```

Normalizing Flow policy on Dense Gridworld environment:
```
TODO
```

Normalizing Flow policy on Sparse Gridworld environment:
```
TODO
```

To run an experiment with a different policy distribution, modify the `--policy` flag.

## References
* Implementation of SAC based on [PyTorch SAC](https://github.com/pranz24/pytorch-soft-actor-critic).