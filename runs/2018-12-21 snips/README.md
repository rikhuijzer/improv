```
corpus = snips2017
num_train_steps = 1500
train_batch_size = 32
n_train_examples = 2099
n_epochs = 1500 * 32 / 2099 = 22.9
eval_size = ...
```

## joint
intents weighted f1: 0.979

entities weighted f1: 0.859

## intent only
intents weighted f1: 0.036

intents weighted f1: 0.0  # second run, all predictions <0>

## entity only
entities weighted f1: 0.837
