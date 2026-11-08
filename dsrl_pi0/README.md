# dsrl_pi0

This folder is a copy of DSRL, https://github.com/nakamotoo/dsrl_pi0, at commit `7f48937`.

- The MIT license of the upstream code is in `LICENSE`. We did not change it.
- Each of our edits is marked in the code with a comment that starts "Changed from upstream".
- We removed the upstream files that the experiments do not use.
- The edits add the task and checkpoint-schedule flags, write `variant.json` and the
  practice-episode log, keep the W&B log in the run folder, and remove options and branches
  that no run reaches. No edit changes the rewards, the critic or the actor.

Cite DSRL as:

```
@article{wagenmaker2025steering,
  author    = {Andrew Wagenmaker and Mitsuhiko Nakamoto and Yunchu Zhang and Seohong Park and Waleed Yagoub and Anusha Nagabandi and Abhishek Gupta and Sergey Levine},
  title     = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```
