"""Collect LIBERO episodes with pi0 steered by the DSRL noise policy, train it and evaluate it."""

# Changed from upstream 7f48937: json and os for the practice-episode log.
import json
import math
import os

import jax
import numpy as np
import PIL
import wandb
from openpi_client import image_tools
from tqdm import tqdm


def _quat2axisangle(quat):
    """Convert a quaternion to an axis-angle vector, as in robosuite.

    Copied from robosuite/utils/transform_utils.py at https://github.com/ARISE-Initiative/robosuite.
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def obs_to_img(obs, variant):
    """Convert a raw observation to the resized image for the DSRL actor and critic."""
    # Changed from upstream 7f48937: the ALOHA branch is removed; every run trains a LIBERO task.
    if variant.env == "libero":
        curr_image = obs["agentview_image"][::-1, ::-1]
    else:
        raise NotImplementedError()
    if variant.resize_image > 0:
        curr_image = np.array(
            PIL.Image.fromarray(curr_image).resize((variant.resize_image, variant.resize_image))
        )
    return curr_image


def obs_to_pi_zero_input(obs, variant):
    """Convert a raw observation to the pi0 input: two images, the state and the prompt."""
    # Changed from upstream 7f48937: the ALOHA branch is removed; every run trains a LIBERO task.
    if variant.env == "libero":
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224))

        obs_pi_zero = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ),
            "prompt": str(variant.task_description),
        }
    else:
        raise NotImplementedError()
    return obs_pi_zero


def obs_to_qpos(obs, variant):
    """Return the end-effector position, axis-angle rotation and gripper state."""
    # Changed from upstream 7f48937: the ALOHA branch is removed; every run trains a LIBERO task.
    if variant.env == "libero":
        qpos = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
    else:
        raise NotImplementedError()
    return qpos


# Changed from upstream 7f48937: the perform_control_evals flag, always True, is removed.
def trajwise_alternating_training_loop(
    variant,
    agent,
    env,
    eval_env,
    online_replay_buffer,
    replay_buffer,
    wandb_logger,
    shard_fn=None,
    agent_dp=None,
):
    """Alternate one practice episode with gradient steps until variant.max_steps."""
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    total_env_steps = 0
    i = 0
    wandb_logger.log({"num_online_samples": 0}, step=i)
    wandb_logger.log({"num_online_trajs": 0}, step=i)
    wandb_logger.log({"env_steps": 0}, step=i)

    # Changed from upstream 7f48937: log the outcome of every practice episode (upstream logs one
    # per log interval) and save checkpoints at an explicit schedule. Training itself is unchanged.
    # ckpt_selection/layout.py PRACTICE_LOG_NAME reads this file name; keep the two in sync.
    practice_log_path = os.path.join(variant.outputdir, "practice_episodes.jsonl")
    open(practice_log_path, "w").close()
    checkpoint_steps = set(variant.checkpoint_steps)

    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:
            traj = collect_traj(variant, agent, env, i, agent_dp)
            traj_id = online_replay_buffer._traj_counter
            # Changed from upstream 7f48937: one line per practice episode; grad_step is the step
            # of the policy that collected it.
            with open(practice_log_path, "a") as practice_log:
                practice_log.write(
                    json.dumps(
                        {
                            "grad_step": int(i),
                            "episode": int(traj_id),
                            "success": int(traj["is_success"]),
                        }
                    )
                    + "\n"
                )
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            total_env_steps += traj["env_steps"]
            print("online buffer timesteps length:", len(online_replay_buffer))
            print("online buffer num traj:", traj_id + 1)
            print("total env steps:", total_env_steps)

            # Changed from upstream 7f48937: the num_online_gradsteps_batch override, which no
            # run sets, is removed.
            num_gradsteps = len(traj["rewards"]) * variant.multi_grad_step

            if len(online_replay_buffer) > variant.start_online_updates:
                for _ in range(num_gradsteps):
                    # perform first visualization before updating
                    if i == 0:
                        print("performing evaluation for initial checkpoint")
                        perform_control_eval(eval_env, i, variant, wandb_logger, agent_dp)
                        # Changed from upstream 7f48937: no hasattr guard (PixelSACLearner has
                        # perform_eval) and no unused arguments.
                        agent.perform_eval(i, wandb_logger, replay_buffer)

                    # online perform update once we have some amount of online trajs
                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch)

                    pbar.update()
                    i += 1

                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        # Changed from upstream 7f48937: only scalar statistics are logged (the
                        # histogram branch for non-scalar ones is removed with log_histogram).
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f"training/{k}": v}, step=i)
                        wandb_logger.log(
                            {
                                "replay_buffer_size": len(online_replay_buffer),
                                "episode_return (exploration)": traj["episode_return"],
                                "is_success (exploration)": int(traj["is_success"]),
                            },
                            i,
                        )

                    # Changed from upstream 7f48937: the periodic evaluation every eval_interval
                    # steps is removed; the step-0 evaluation above stays.

                    # Changed from upstream 7f48937: save exactly at the --checkpoint_steps schedule
                    # and keep every checkpoint (keep_every_n_steps=1); the upstream
                    # checkpoint_interval rule is removed.
                    if i in checkpoint_steps:
                        agent.save_checkpoint(variant.outputdir, i, 1)


def add_online_data_to_buffer(variant, traj, online_replay_buffer):
    """Insert each noise-chunk transition of an episode into the online replay buffer."""
    discount_horizon = variant.query_freq
    actions = np.array(traj["actions"])  # (T, chunk_size, action_dim )
    episode_len = len(actions)
    rewards = np.array(traj["rewards"])
    masks = np.array(traj["masks"])

    for t in range(episode_len):
        obs = traj["observations"][t]
        next_obs = traj["observations"][t + 1]
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop("state", None)
            next_obs.pop("state", None)

        # Changed from upstream 7f48937: next_actions is not stored (ReplayBuffer no longer
        # keeps it; no update reads it).
        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount**discount_horizon,
        )
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()


def collect_traj(variant, agent, env, i, agent_dp=None):
    """Run one practice episode and return its transitions with sparse -1/0 rewards."""
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward

    agent._rng, rng = jax.random.split(agent._rng)

    # Changed from upstream 7f48937: the ALOHA reset and step branches are removed.
    obs = env.reset()

    # Changed from upstream 7f48937: the episode's image list, which no caller reads, is removed.
    rewards = []
    action_list = []
    obs_list = []

    for t in tqdm(range(max_timesteps)):
        curr_image = obs_to_img(obs, variant)

        qpos = obs_to_qpos(obs, variant)

        if variant.add_states:
            obs_dict = {
                "pixels": curr_image[np.newaxis, ..., np.newaxis],
                "state": qpos[np.newaxis, ..., np.newaxis],
            }
        else:
            obs_dict = {
                "pixels": curr_image[np.newaxis, ..., np.newaxis],
            }

        if t % query_frequency == 0:
            assert agent_dp is not None
            # we then use the noise to sample the action from diffusion model
            rng, key = jax.random.split(rng)
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)
            if i == 0:
                # for initial round of data collection, we sample from standard gaussian noise
                noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                noise_repeat = jax.numpy.repeat(noise[:, -1:, :], 50 - noise.shape[1], axis=1)
                noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
                actions_noise = noise[0, : agent.action_chunk_shape[0], :]
            else:
                # sac agent predicts the noise for diffusion model
                actions_noise = agent.sample_actions(obs_dict)
                actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                noise = np.repeat(actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]

            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
            action_list.append(actions_noise)
            obs_list.append(obs_dict)

        action_t = actions[t % query_frequency]
        obs, reward, done, _ = env.step(action_t)

        rewards.append(reward)
        if done:
            break

    # add last observation
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    obs_dict = {
        "pixels": curr_image[np.newaxis, ..., np.newaxis],
        "state": qpos[np.newaxis, ..., np.newaxis],
    }
    obs_list.append(obs_dict)

    # per episode
    rewards = np.array(rewards)
    # Elementwise numpy mask, so `is not None` would change the result.
    episode_return = np.sum(rewards[rewards != None])  # noqa: E711
    is_success = reward == env_max_reward
    print(f"Rollout Done: {episode_return=}, Success: {is_success}")

    # We use sparse -1/0 reward to train the SAC agent.
    if is_success:
        query_steps = len(action_list)
        rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
        masks = np.concatenate([np.ones(query_steps - 1), [0]])
    else:
        query_steps = len(action_list)
        rewards = -np.ones(query_steps)
        masks = np.ones(query_steps)

    return {
        "observations": obs_list,
        "actions": action_list,
        "rewards": rewards,
        "masks": masks,
        "is_success": is_success,
        "episode_return": episode_return,
        "env_steps": t + 1,
    }


# Changed from upstream 7f48937: the agent parameter, unused once the agent branch is removed,
# is dropped.
def perform_control_eval(env, i, variant, wandb_logger, agent_dp=None):
    """Evaluate the base policy for variant.eval_episodes episodes and log the results."""
    query_frequency = variant.query_freq
    # Changed from upstream 7f48937: the debug print of the query frequency is removed.
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    episode_returns = []
    highest_rewards = []
    success_rates = []
    episode_lens = []

    rng = jax.random.PRNGKey(variant.seed + 456)

    for rollout_id in range(variant.eval_episodes):
        # Changed from upstream 7f48937: the ALOHA reset and step branches are removed.
        obs = env.reset()

        image_list = []  # for visualization
        rewards = []

        for t in tqdm(range(max_timesteps)):
            curr_image = obs_to_img(obs, variant)

            if t % query_frequency == 0:
                rng, key = jax.random.split(rng)
                assert agent_dp is not None

                obs_pi_zero = obs_to_pi_zero_input(obs, variant)

                # Changed from upstream 7f48937: the only call is at step 0, so only the base
                # policy is evaluated, with standard gaussian noise. The agent branch and the
                # actor observation it read are removed.
                noise = jax.random.normal(rng, (1, 50, 32))

                actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]

            action_t = actions[t % query_frequency]
            obs, reward, done, _ = env.step(action_t)

            rewards.append(reward)
            image_list.append(curr_image)
            if done:
                break

        # per episode
        episode_lens.append(t + 1)
        rewards = np.array(rewards)
        episode_return = np.sum(rewards)
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        is_success = reward == env_max_reward
        success_rates.append(is_success)

        print(f"Rollout {rollout_id} : {episode_return=}, Success: {is_success}")
        video = np.stack(image_list).transpose(0, 3, 1, 2)
        wandb_logger.log({f"eval_video/{rollout_id}": wandb.Video(video, fps=50)}, step=i)

    success_rate = np.mean(np.array(success_rates))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    wandb_logger.log({"evaluation/avg_return": avg_return}, step=i)
    wandb_logger.log({"evaluation/success_rate": success_rate}, step=i)
    wandb_logger.log({"evaluation/avg_episode_len": avg_episode_len}, step=i)
    for reward_threshold in range(env_max_reward + 1):
        num_at_or_above = (np.array(highest_rewards) >= reward_threshold).sum()
        rate_at_or_above = num_at_or_above / variant.eval_episodes
        wandb_logger.log({f"evaluation/Reward >= {reward_threshold}": rate_at_or_above}, step=i)
        summary_str += (
            f"Reward >= {reward_threshold}: {num_at_or_above}/{variant.eval_episodes}"
            f" = {rate_at_or_above * 100}%\n"
        )

    print(summary_str)


# Changed from upstream 7f48937: the unused variant parameter is removed.
def make_multiple_value_reward_visulizations(agent, i, replay_buffer, wandb_logger):
    """Log the value and reward plots of three random trajectories from the replay buffer."""
    trajs = replay_buffer.get_random_trajs(3)
    images = agent.make_value_reward_visulization(trajs)
    wandb_logger.log({"reward_value_images": wandb.Image(images)}, step=i)
