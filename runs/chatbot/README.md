```
corpus = chatbot
num_train_steps = 250
train_batch_size = 16
n_train_examples = 99
n_eval_examples = 105
```

English model cannot interpret umlaut. Tokens with it will be [UNK]. 

## joint
intents weighted f1: 0.981

entities weighted f1: 0.792

## intent only
intents weighted f1: 0.991

intents weighted f1: 0.991  # vanilla model

## entity only
entities weighted f1: 0.746

## rasa spacy
intents weighted f1: 0.981
