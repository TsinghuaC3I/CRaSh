import argparse
from transformers import SchedulerType, MODEL_MAPPING

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")

    #### for dataset
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help=
        "The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.")
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.")
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        type=int,
        help=
        "The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=512,
        help=
        ("Optional input sequence length after tokenization. The training dataset will be truncated in block of"
         " this size for training. Default to the model max input length for single sentence inputs (take into"
         " account special tokens)."),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--no_keep_linebreaks",
                        action="store_true",
                        help="Do not keep line breaks when using TXT files.")

    #### for model and tokenizer
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help=
        "If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument('--optimizer',
                        type=str,
                        default='adamw',
                        help='Optimizer to use. Can be adamw or sgd',
                        choices=['adamw', 'sgd'])
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.0,
                        help="Weight decay to use.")
    parser.add_argument("--momentum",
                        type=float,
                        default=0.9,
                        help="Momentum to use for sgd optimizer.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help=
        "Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the final model.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help=
        "The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token",
                        type=str,
                        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help=
        "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=
        ('The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
         ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
         "Only applicable when `--with_tracking` is passed."),
    )
    parser.add_argument('--no_save_model',
                        action='store_true',
                        help='Whether or not to save the model.')
    parser.add_argument('--train_tokenized_dataset',
                        type=str,
                        default=None,
                        help='Path to the tokenized training dataset.')
    parser.add_argument('--val_tokenized_dataset',
                        type=str,
                        default=None,
                        help='Path to the tokenized validation dataset.')
    parser.add_argument(
        "--train_num_samples",
        type=int,
        default=None,
        help="The number of samples to use for training set.",
    )
    parser.add_argument(
        "--validation_num_samples",
        type=int,
        default=None,
        help="The number of samples to use for validation set.",
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=200,
    )
    parser.add_argument(
        "--save_for_eval",
        action='store_true',
        help="Whether to save the model for evaluation.",
    )
    parser.add_argument('--restart_training',
                        action='store_true',
                        help='Whether to restart training of all dataset.')

    parser.add_argument('--train_module',
                        type=str,
                        default='student',
                        help='Part of the model to train.',
                        choices=['student', 'adapter', 'all'])
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=1.0,
                        help='Max gradient norm.')

    # vit
    parser.add_argument("--train_dir",
                        type=str,
                        default=None,
                        help="A folder containing the training data.")
    parser.add_argument("--validation_dir",
                        type=str,
                        default=None,
                        help="A folder containing the validation data.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=
        ("For debugging purposes or quicker training, truncate the number of training examples to this "
         "value if set."),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=
        ("For debugging purposes or quicker training, truncate the number of evaluation examples to this "
         "value if set."),
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.15,
        help="Percent to split off of train for validation",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help=
        "Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=
        ("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
         " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
         ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=12,
    )

    parser.add_argument(
        '--train_lm_head',
        action='store_true',
    )

    parser.add_argument('--load_adapter',
                        type=str,
                        default=None,
                        help='Path to the student model')

    parser.add_argument(
        '--tasks',
        type=str,
        default='piqa',
        help='Evaluation tasks',
    )

    parser.add_argument(
        '--use_adapter',
        action='store_true',
    )

    parser.add_argument(
        '--use_lora',
        action='store_true',
    )

    parser.add_argument(
        '--use_bitfit',
        action='store_true',
    )

    parser.add_argument(
        '--lora_rank',
        type=int,
        default=4,
        help='Rank of the LoRA matrix',
    )

    parser.add_argument(
        '--lora_alpha',
        type=float,
        default=32,
        help='Alpha of the LoRA matrix',
    )

    parser.add_argument(
        '--adapter_size',
        type=int,
        default=64,
        help='Size of the adapter',
    )

    #### trainable layers
    parser.add_argument('--student_strategy',
                        type=str,
                        default="choose",
                        choices=['choose', 'monarch', 'lowrank_svd', 'repeat'],
                        help='how to get student layers')
    parser.add_argument(
        '--student_layers',
        type=str,
        default=None,  #"2,5,7,10,13,16,18,21",
        help='student layer index in the format of 0,1,2,3,4')
    parser.add_argument(
        '--train_layers',
        type=str,
        default=None,  #"0,1,22,23",
        help='train layer index in the format of 0,1,2,3,4')
    parser.add_argument('--lowrank_svd_strategy',
                        type=str,
                        default="topk",
                        choices=['topk', 'sampling', 'lowrank'],
                        help='topk or sampling to get lowrank svd rank')
    parser.add_argument('--lowrank_svd_topk',
                        type=str,
                        default="16",
                        help='topk for lowrank')
    parser.add_argument(
        '--only_context',
        action='store_true',
    )
    parser.add_argument(
        '--only_repeat_non_trainable_layers',
        action='store_true',
    )
    parser.add_argument(
        '--zero_shot_eval',
        action='store_true',
    )
    parser.add_argument(
        '--load_student',
        action='store_true',
    )

    #### for linear mode connectivity
    parser.add_argument('--interpolate_adapter',
                        action='store_true',
                        help='Interpolate adapter with origin weights')
    parser.add_argument('--interpolate_all',
                        action='store_true',
                        help='Interpolate adapter with origin weights')
    parser.add_argument(
        '--interpolate_coef',
        type=float,
        default=None,
        help='interpolate coefficient between adapter and origin weights',
    )
    parser.add_argument('--load_interp_adapter',
                        type=str,
                        default=None,
                        help='Path to the adapter model')
    parser.add_argument('--load_interp_student',
                        type=str,
                        default=None,
                        help='Path to the student model')
    parser.add_argument(
        '--interp_student_l_pad',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--interp_student_r_pad',
        type=int,
        default=0,
    )


    # for 2d loss surface
    parser.add_argument(
        "--assist_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument('--assist_adapter',
                        type=str,
                        default=None,
                        help='Path to the student model')
    parser.add_argument('--assist_student',
                        type=str,
                        default=None,
                        help='Path to the student model')
    parser.add_argument(
        '--assist_student_l_pad',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--assist_student_r_pad',
        type=int,
        default=0,
    )

    #### for visualization
    parser.add_argument('--cka_minibatch',
                        type=int,
                        default=1,
                        help='the number of minibatch to compute cka')
    parser.add_argument(
        '--swap_to_cpu',
        action='store_true',
    )
    parser.add_argument(
        '--split_layer_by_layer',
        action='store_true',
    )
    parser.add_argument('--visual_type',
                        type=str,
                        default="layer_cka",
                        choices=['layer_cka', 'weight_l2', 'layer_var'],
                        help="visualization type")
    parser.add_argument('--additional_note',
                        type=str,
                        default=None,
                        help="Additional note")
    parser.add_argument(
        "--another_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument('--pooling',
                        type=str,
                        default="mean",
                        choices=["mean", "max", "min", "last", "none", "mean_all", "max_nonpad", "mean_weighted"],
                        help="how to pool the embeddings")
    parser.add_argument('--title',
                        type=str,
                        default=None,
                        help="title for visualization")
    parser.add_argument('--xlabel',
                        type=str,
                        default=None,
                        help="xlabel for visualization")
    parser.add_argument('--ylabel',
                        type=str,
                        default=None,
                        help="ylabel for visualization")
    parser.add_argument('--datasets_to_cluster',
                        type=str,
                        default=None,
                        help="datasets to cluster split by ,")

    parser.add_argument
    args = parser.parse_args()

    return args