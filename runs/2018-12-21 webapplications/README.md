```
corpus = webapplications
num_train_steps = 250
train_batch_size = 32
n_train_examples = 29
n_epochs = 250 * 32 / 29 = 276
n_eval_examples = 58
```

## joint
intents weighted f1: 0.658

entities weighted f1: 0.819

### steps 1000 with train batch size 8
intents weighted f1: 0.532

entities weighted f1: 0.809

### step 1500 with train batch size 8
intents weighted f1: 0.679

entities weighted f1: 0.837

## intent only
intents weighted f1: 0.0

intents weighted f1: 0.116  # vanilla BERT code

intents weighted f1: 0.724  # vanilla BERT code 1000 train steps batch size 8

## entity only
entities weighted f1: 0.79

## rasa-spacy
intents weighted f1: 0.674
