# pytoch-dqn

**Work in Progress**: If you are willing to help with the implementation, please check <https://discuss.pytorch.org/t/help-for-dqn-implementation-of-the-paper/1122>.

This project is pytorch implementation of [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) and I also plan to implement the following ones:

- [ ] [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [ ] [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

# Credit

This project reuses most of the code in <https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3>

# Requirements

- python 3.5
- [gym](https://github.com/openai/gym#installation) (built from source)
- [pytorch](https://github.com/pytorch/pytorch#from-source) (built from source)

# Usage

To train a model:

```
$ python main.py

# To train the model using ram not raw images

$ python ram.py
```

The model is defined in `dqn_model.py`

The algorithm is defined in `dqn_learn.py`

The running script and hyper-parameters are defined in `main.py`

**Note**: Adjusting hyper-parameters using command line is coming soon in the future once I get everything working.
