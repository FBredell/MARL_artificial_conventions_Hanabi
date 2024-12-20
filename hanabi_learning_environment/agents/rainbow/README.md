# Rainbow with augmented convention-action space agents for the Hanabi Learning Environment

## Instructions - Basics

The Rainbow agent is derived from the
[Dopamine framework](https://github.com/google/dopamine) which is based on
Tensorflow. We recommend you consult the
[Tensorflow documentation](https://www.tensorflow.org/install)
for additional details.

To run the agent, some dependencies need to be pre-installed. If you don't have
access to a GPU, then replace `tensorflow-gpu` with `tensorflow` in the line
below
(see [Tensorflow instructions](https://www.tensorflow.org/install/install_linux)
for details).

This assumes you already installed the learning environment as detailed in the root README.

```
pip install absl-py gin-config tensorflow-gpu==1.15.0 numpy
```

If you would prefer to not use the GPU, you may install tensorflow instead
of tensorflow-gpu and set `RainbowAgent.tf_device = '/cpu:*'` in
`configs/hanabi_rainbow.gin`.

The entry point to run a Rainbow agent on the Hanabi environment is `train.py`.
Assuming you are running from the agent directory `hanabi_learning_environment/agents/rainbow`,

```
python -um train \
  --base_dir=/tmp/hanabi_rainbow \
  --gin_files='configs/hanabi_rainbow.gin'
```

The `--base_dir` argument must be provided.

To get finer-grained information about the training process, you can adjust the
experiment parameters in `configs/hanabi_rainbow.gin` in particular by reducing
`Runner.training_steps` and `Runner.evaluation_steps`, which together determine
the total number of steps needed to complete an iteration. This is useful if you
want to inspect log files or checkpoints, which are generated at the end of each
iteration.

More generally, most parameters are easily configured using the
[gin configuration framework](https://github.com/google/gin-config).

## Instructions - Conventions

After editing and setting the needed parameters for full Hanabi, the convention agents should automatically set the needed conventions for the number of players specified. To switch to Small Hanabi conventions, the indicated code sections found within the train.py must be uncommented, and the preffered convention encoder must be selected. Although the Full Hanabi conventions are based on official Hanabi conventions by the [H-Group](https://hanabi.github.io/), the Small Hanabi conventions were developed in house, and we do not claim that they are the most optimal conventions. 

We have also included one additional attempts at convention experiments, with partical focus on transfer learning, however these are experimental and did not result in noticable performance gains.
