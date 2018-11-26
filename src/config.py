from typing import NamedTuple
from pathlib import Path
from src.utils import get_project_root


HParams = NamedTuple('Params', [
    ('data_dir', Path),
    ('bert_config_file', Path),
    ('task_name', str),
    ('vocab_file', Path),
    ('output_dir', str),  # using string to easy TPU location definition
    ('init_checkpoint', Path),  # Initial checkpoint (usually from a pre-trained BERT model).
    ('do_lower_case', bool),  # should be True for uncased and False otherwise
    ('max_seq_length', int),  # Total input sentence length after WordPiece tokenization. Truncated and padded to match.
    ('do_train_eval', bool),  # Since TPU does not support TensorBoard we run a loop to get statistics
    ('do_train', bool),
    ('do_eval', bool),
    ('do_predict', bool),
    ('train_batch_size', int),
    ('eval_batch_size', int),
    ('predict_batch_size', int),
    ('learning_rate', float),
    ('num_train_epochs', float),
    ('warmup_proportion', float),
    ('save_checkpoints_steps', int),  # how often to save the model checkpoint
    ('iterations_per_loop', int),  # how many steps to make in each estimator call
    ('use_tpu', bool),
    ('tpu_name', str),  # Either the name used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470
    ('tpu_zone', str),  # [Optional] GCE zone for Cloud TPU, if not specified automated detection is attempted
    ('gcp_project', str),  # [Optional] Project name Cloud TPU, if not specified automated detection is attempted
    ('master', str),  # [Optional] TensorFlow master URL
    ('num_tpu_cores', int)  # Only used if use_tpu is True
])


def get_debug_hparams() -> HParams:
    """Parameters for lightweight BERT execution for debug purposes."""
    task_name = 'askubuntu'
    bert_model = 'uncased_L-12_H-768_A-12'
    bert_pretrained_dir = Path.home() / 'Downloads' / bert_model
    output_dir_name = 'debug'

    return HParams(
        data_dir=get_project_root() / 'data' / task_name,
        bert_config_file=bert_pretrained_dir / 'bert_config.json',
        task_name=task_name,
        vocab_file=bert_pretrained_dir / 'vocab.txt',
        output_dir=str(get_project_root() / 'tmp' / task_name / output_dir_name),
        init_checkpoint=bert_pretrained_dir / 'bert_model.ckpt',
        do_lower_case=bert_model.startswith('uncased'),
        max_seq_length=128,
        do_train_eval=False,
        do_train=False,
        do_eval=False,
        do_predict=False,
        train_batch_size=1,
        eval_batch_size=8,
        predict_batch_size=8,
        learning_rate=2e-5,
        num_train_epochs=2.0,
        warmup_proportion=0.1,
        save_checkpoints_steps=1000,
        iterations_per_loop=1000,
        use_tpu=False,
        tpu_name='',  # is used as tpu_address in Colab script
        tpu_zone=None,
        gcp_project=None,
        master=None,
        num_tpu_cores=8
    )


def get_hparams() -> HParams:
    """Parameters for running BERT. Defaults can be found on Github."""
    return HParams(
        data_dir=None,
        bert_config_file=None,
        task_name=None,
        vocab_file=None,
        output_dir=None,
        init_checkpoint=None,
        do_lower_case=True,
        max_seq_length=128,
        do_train_eval=False,
        do_train=False,
        do_eval=False,
        do_predict=False,
        train_batch_size=32,
        eval_batch_size=8,
        predict_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=3.0,
        warmup_proportion=0.1,
        save_checkpoints_steps=1000,
        iterations_per_loop=1000,
        use_tpu=False,
        tpu_name=None,
        tpu_zone=None,
        gcp_project=None,
        master=None,
        num_tpu_cores=8
    )
