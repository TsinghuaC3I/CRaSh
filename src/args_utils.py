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
    parser.add_argument('--kd_weight',
                        type=float,
                        default=0.0,
                        help='Weight of the knowledge distillation loss.')
    parser.add_argument('--lm_weight',
                        type=float,
                        default=1.0,
                        help='Weight of the knowledge distillation loss.')
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

    parser.add_argument('--num_student_layers',
                        type=int,
                        default=None,
                        help='Number of layers in the student model.')

    parser.add_argument('--load_student',
                        type=str,
                        default=None,
                        help='Path to the student model')

    parser.add_argument(
        '--student_l_pad',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--student_r_pad',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--fixed_adapter_index',
        type=str,
        default="0,1,12,23",
    )
    parser.add_argument(
        '--fixed_student_index',
        type=str,
        default="3,6,9,11,13,15,19,22",
    )

    parser.add_argument('--student_layer_selection_strategy',
                        type=str,
                        default='uniform',
                        help='Layer selection strategy',
                        choices=[
                            'uniform', 'random', 'changes', 'normal',
                            'left_normal', 'right_normal', 'inverse_normal',
                            "fixed", "all"
                        ])
    parser.add_argument(
        '--uniform_deviation',
        type=int,
        default=0,
        choices=[-1, 0, 1],
    )

    parser.add_argument('--uniform_percent', type=float, default=None)

    parser.add_argument(
        '--normal_variance',
        type=float,
        default=1 / 2.5,
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

    parser.add_argument('--magnitude_pruning_ratio',
                        type=float,
                        default=0.0,
                        help='Magnitude pruning ratio.')

    parser.add_argument('--weight_quantization_bits',
                        type=int,
                        default=None,
                        help='Weight quantization bits.')
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss")

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
        '--freeze_bottom',
        action='store_true',
    )
    parser.add_argument(
        '--no_teacher',
        action='store_true',
    )

    parser.add_argument(
        '--classifier_lr_multiplier',
        type=float,
        default=1.0,
    )

    parser.add_argument(
        '--select_by_kd',
        action='store_true',
    )

    parser.add_argument(
        '--use_pt_imagefolder',
        action='store_true',
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
    parser.add_argument('--save_module',
                        type=str,
                        default='student',
                        choices=['student', 'adapter', 'all'])

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
    parser.add_argument(
        '--rewind_layer_id',
        type=int,
        default=None,
        help='layer to rewind',
    )
    parser.add_argument(
        '--delete_layer_id',
        type=int,
        default=None,
        help='layer to delete',
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
    parser.add_argument('--load_assist_adapter_to_init',
                        type=str,
                        default=None,
                        help='Path to the assist adapter model for initialization')
    parser.add_argument('--assist_student',
                        type=str,
                        default=None,
                        help='Path to the student model')
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
    parser.add_argument(
        '--point_strs',
        type=str,
        default="start,end1,end2",
        help='init, full fine-tune, assist')
    parser.add_argument(
        '--assist_student_l_pad',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--load_finetuned_for_assist',
        action='store_true',
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
    parser.add_argument('--num_keep_layers',
                        type=int,
                        default=8,
                        help='the number of layers to keep')
    parser.add_argument('--num_layer_cluster',
                        type=int,
                        default=4,
                        help='the number of cluster')
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
                        choices=[
                            'layer_cka', 'weight_l2', 'layer_var',
                            'layer_iter_drop', 'layer_iter_cluster',
                            'pair_layer_replace', 'logits_len',
                            'evaluate_layer_importance', "datasets_cluster", "show_datasets", "tuned_lens_similarity"
                        ],
                        help="visualization type")
    parser.add_argument(
        '--visual_forward_type',
        type=str,
        default="stacking",
        choices=['stacking', 'single'],
        help="Forward type for visualization: stacking or single")
    parser.add_argument(
        '--similarity_type',
        type=str,
        default="js_div",
        choices=['wasserstein_distance', 'hellinger_distance', 'js_div', "kl_div", 'cosine_similarity'],
        help="similarity type")
    parser.add_argument('--wikitext_range',
                    type=str,
                    default="32-64",
                    help='the range of wikitext length to visualize')
    # parser.add_argument('--eval_layer_type',
    #                     type=str,
    #                     default="delete",
    #                     choices=['delete', 'rewind'],
    #                     help="how to evaluate layer importance")
    parser.add_argument('--fig_dir',
                        type=str,
                        default=None,
                        help="Directory to save figures")
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
    parser.add_argument('--only_context',
                        action='store_true',
                        help="whether only compute context embeddings")
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