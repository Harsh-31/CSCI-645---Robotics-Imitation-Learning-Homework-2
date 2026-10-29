"""Sanity check from the HW README: random-action rollout -> MP4.

Usage: MUJOCO_GL=egl python code/sanity_random_rollout.py
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import common  # noqa: E402

env = common.load_env()
print("obs size", env.observation_size, "action size", env.action_size,
      "dt", env.dt, "episode length", env._config.episode_length)
print(common.run_metadata())
reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)

state = reset_fn(jax.random.PRNGKey(0))
trajectory = [state]
rng = jax.random.PRNGKey(42)
action = jnp.zeros(env.action_size)
for i in range(200):
  if i % 10 == 0:
    rng, action_rng = jax.random.split(rng)
    action = jax.random.uniform(action_rng, shape=(env.action_size,),
                                minval=-1.0, maxval=1.0)
  state = step_fn(state, action)
  trajectory.append(state)

frames = env.render(trajectory[::2], height=480, width=640)
out = common.ROOT / "videos" / "sanity_random_actions.mp4"
common.write_video(out, frames, env)
print("Saved", out)
