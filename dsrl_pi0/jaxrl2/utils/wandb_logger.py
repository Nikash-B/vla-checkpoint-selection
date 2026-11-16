"""Name a training run and write its W&B log of training scalars."""

import datetime
import os

import dateutil.tz
import wandb


# Changed from upstream 7f48937: the exp_id parameter, always 0, is removed; the name keeps its
# "0000" field, so run names are unchanged.
def create_exp_name(exp_prefix, seed=0):
    """Return a run name built from the prefix, a local timestamp, the run id 0000 and the seed."""
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    return "%s_%s_0000--s-%d" % (exp_prefix, timestamp, seed)


class WandBLogger:
    """Write training scalars to a W&B run in output_dir/experiment_id.

    The run is offline when WANDB_MODE=offline, as the run scripts set.
    """

    def __init__(
        self, wandb_logging, variant, project, experiment_id, output_dir=None, group_name=""
    ):
        """Create the run folder and, if wandb_logging is set, start the W&B run."""
        self.wandb_logging = wandb_logging
        output_dir = os.path.join(output_dir, experiment_id)
        os.makedirs(output_dir, exist_ok=True)
        if wandb_logging:
            print("wandb using experimentid: ", experiment_id)
            print("wandb using project: ", project)
            print("wandb using group: ", group_name)
            # Changed from upstream 7f48937: WANDB_MODE is not forced to "run" (the run scripts
            # set offline), and the unused self.output_dir is not kept.
            wandb.init(
                config=variant,
                project=project,
                dir=output_dir,
                id=experiment_id,
                settings=wandb.Settings(start_method="thread"),
                group=group_name,
            )

    def log(self, *args, **kwargs):
        """Pass a dict of logged values and its step to wandb.log."""
        if self.wandb_logging:
            wandb.log(*args, **kwargs)
