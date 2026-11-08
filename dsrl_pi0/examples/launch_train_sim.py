"""Parse the command line and start one DSRL training run on a LIBERO task."""

import argparse

from jaxrl2.utils.launch_util import parse_training_args

from examples.train_sim import main


# Changed from upstream 7f48937: parse the explicit checkpoint schedule given on the command line.
def parse_step_list(text):
    """Return the integer steps of a comma list such as '1000,2500'."""
    return [int(step) for step in text.split(",") if step.strip()]


def launch():
    """Parse the command line and run training with the fixed DSRL hyperparameters."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, help="Random seed.", type=int)
    # Changed from upstream 7f48937: --launch_group_id is removed; the W&B group is the prefix.
    parser.add_argument(
        "--eval_episodes", default=10, help="Number of episodes used for evaluation.", type=int
    )
    parser.add_argument("--env", default="libero", help="Name of the environment.")
    parser.add_argument("--log_interval", default=1000, help="Logging interval.", type=int)
    # Changed from upstream 7f48937: --eval_interval is removed with the periodic evaluation, and
    # --checkpoint_interval with the interval rule (--checkpoint_steps below replaces it).
    parser.add_argument("--batch_size", default=16, help="Mini batch size.", type=int)
    parser.add_argument("--max_steps", default=int(1e6), help="Number of training steps.", type=int)
    parser.add_argument(
        "--add_states",
        default=1,
        help="Whether to add low-dim states to the observations.",
        type=int,
    )
    parser.add_argument("--wandb_project", default="cql_sim_online", help="Wandb project.")
    parser.add_argument(
        "--start_online_updates",
        default=1000,
        help="Number of steps to collect before starting online updates.",
        type=int,
    )
    # Changed from upstream 7f48937: --prefix is required (upstream drew a random one when empty),
    # and --suffix, which no run sets, is removed.
    parser.add_argument("--prefix", required=True, help="Run name prefix, also the W&B group.")
    parser.add_argument(
        "--multi_grad_step",
        default=1,
        help="Number of gradient steps to take per environment step, aka UTD.",
        type=int,
    )
    parser.add_argument(
        "--resize_image",
        default=-1,
        help="Image size after resizing; -1 keeps the original size.",
        type=int,
    )
    parser.add_argument("--query_freq", default=-1, help="Query frequency.", type=int)
    # Changed from upstream 7f48937: task, checkpoint schedule and output folder are required
    # flags, so one code copy trains every run of the paper (upstream fixes libero_90 task 57, a
    # checkpoint interval and a dated folder).
    parser.add_argument("--task_suite", required=True, help="LIBERO suite, e.g. libero_10.")
    parser.add_argument("--task_id", required=True, help="Task id within the suite.", type=int)
    parser.add_argument(
        "--checkpoint_steps",
        required=True,
        type=parse_step_list,
        help="Comma list of the gradient steps at which to save a checkpoint.",
    )
    parser.add_argument("--output_dir", required=True, help="Run folder.")

    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(128, 128, 128),
        cnn_features=(32, 32, 32, 32),
        cnn_strides=(2, 1, 1, 1),
        cnn_padding="VALID",
        latent_dim=50,
        discount=0.999,
        tau=0.005,
        critic_reduction="mean",
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type="small",
        # Changed from upstream 7f48937: encoder_norm, use_spatial_softmax and
        # softmax_temperature, which only configured the removed ResNet encoders, are removed.
        target_entropy="auto",
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=1,
    )

    variant = parse_training_args(train_args_dict, parser)
    main(variant)


if __name__ == "__main__":
    launch()
