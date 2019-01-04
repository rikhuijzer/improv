```
corpus = askubuntu
num_train_steps = 250
train_batch_size = 32
n_train_examples = 52
n_eval_examples = 108
```

Loss at joint was still 7.8, this might be the cause for intent not yet being learned.
With 154 epochs this is not expected. Dataset too small for BERT it seems.

## joint
intents weighted f1: 0.554

entities weighted f1: 0.826

## intent only
intents weighted f1: 0.857
intents weighted f1: 0.83  # using vanilla BERT
intents weighted f1: 0.197  # second run on vanilla BERT

## entity only
entities weighted f1: 0.81

## rasa spacy
intents weighted f1: 0.834
