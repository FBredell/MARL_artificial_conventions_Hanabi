## Overview 

The code within this repository contains the implimentation of artificial conventions based on Human conventions applied to independant deep Q-learning as well as multi-agent Rainbow within Hanabi. We used the code available at https://github.com/google-deepmind/hanabi-learning-environment as basis for building our method, and have clearly indicated where modifications have been made. The convention encoders found within hanabi_learning_environment/agents/rainbow are our original work, made publically available for future research endeavours, as long as appropriate credit is given.

The full Hanabi conventions implemented in hanabi_learning_environment/agents/rainbow/hanabi_conventions_encoder.py is based on the conventions of the [H-Group](https://hanabi.github.io/), and we strongly recommend familiarising yourself with the basic conventions before looking at our work.  


### Getting started - Hanabi Learning Environment

hanabi\_learning\_environment is a research platform for Hanabi experiments. The file rl\_env.py provides an RL environment using an API similar to OpenAI Gym. A lower level game interface is provided in pyhanabi.py for non-RL methods like Monte Carlo tree search.

Install the learning environment:
```
sudo apt-get install g++            # if you don't already have a CXX compiler
sudo apt-get install python-pip     # if you don't already have pip
pip install .                       # or pip install git+repo_url to install directly from github
```
Run the examples:
```
pip install numpy                   # game_example.py uses numpy
python examples/rl_env_example.py   # Runs RL episodes
python examples/game_example.py     # Plays a game using the lower level interface
```

We have also included an environment.yml file to quickly install all the dependencies using conda. 

### Running the convention agents

We have included conventions for Small Hanabi, as well as Full Hanabi found within hanabi_learning_environment/agents/rainbow. These conventions are also applicable to independant deep Q-learning, and can easily be run by simply changing the algorithm in the config file hanabi_learning_environment/agents/rainbow/configs/hanabi_rainbow.gin. Due to the modifications made to accomodate covnentions, the train.py found within hanabi_learning_environment/agents/rainbow can only run convention agents, and there is no easy way to turn conventions off. To compare against normal agents, we recommend downloading the original https://github.com/google-deepmind/hanabi-learning-environment repository. 

### Citation

To reference our work, and for more information, please see the article detailing our research on Augmenting the action space with conventions to improve multi-agent cooperation in Hanabi:

