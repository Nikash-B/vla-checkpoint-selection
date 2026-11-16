"""Command-line parsing for the training launcher."""

from jaxrl2.utils.general_utils import AttrDict


def parse_training_args(train_args_dict, parser):
    """Add one flag per training argument, parse them, and return the variant."""
    for arg_name, default in train_args_dict.items():
        if type(default) is tuple:
            parser.add_argument("--" + arg_name, nargs="+", default=default, type=type(default[0]))
        elif type(default) is not bool:
            parser.add_argument("--" + arg_name, default=default, type=type(default))
        else:
            parser.add_argument("--" + arg_name, default=int(default), type=int)
    args = parser.parse_args()
    config = {}
    for arg_name in train_args_dict.keys():
        config[arg_name] = getattr(args, arg_name)
    variant = AttrDict(vars(args))
    variant["train_kwargs"] = config
    # Changed from upstream 7f48937: only the variant is returned (the caller dropped the args).
    return variant
