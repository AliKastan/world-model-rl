"""Test script for the Gymnasium environment wrapper.

- Verifies registration and creation of all env variants
- Runs 10 random-policy episodes and reports statistics
- Tests flat observation helper
- Tests rgb_array rendering
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

# Import triggers registration
import env.gym_env  # noqa: F401
import gymnasium


def test_registration() -> None:
    """Verify all four registered environments can be created."""
    for env_id in [
        "ThinkPuzzle-v0",
        "ThinkPuzzle-Easy-v0",
        "ThinkPuzzle-Medium-v0",
        "ThinkPuzzle-Hard-v0",
    ]:
        e = gymnasium.make(env_id)
        obs, info = e.reset(seed=42)
        assert "grid" in obs, f"{env_id}: missing 'grid' in obs"
        assert "agent_pos" in obs, f"{env_id}: missing 'agent_pos'"
        e.close()
        print(f"  {env_id}: OK")


def test_spaces() -> None:
    """Verify observation and action spaces are correct."""
    e = gymnasium.make("ThinkPuzzle-v0")
    obs, _ = e.reset(seed=1)

    assert obs["grid"].shape == (12, 12), f"grid shape: {obs['grid'].shape}"
    assert obs["grid"].dtype == np.int8
    assert obs["agent_pos"].shape == (2,)
    assert obs["agent_pos"].dtype == np.int32
    assert isinstance(obs["agent_dir"], (int, np.integer))
    assert obs["inventory"].shape == (4,)
    assert obs["boxes_on_targets"].shape == (1,)
    assert obs["total_targets"].shape == (1,)
    assert e.action_space.n == 4

    # Check observation is in space
    assert e.observation_space.contains(obs), "obs not in observation_space"

    e.close()
    print("  Spaces: OK")


def test_flat_obs() -> None:
    """Verify flat observation helper."""
    from env.gym_env import get_flat_obs

    e = gymnasium.make("ThinkPuzzle-v0")
    obs, _ = e.reset(seed=1)
    flat = get_flat_obs(obs)

    assert flat.ndim == 1
    assert flat.dtype == np.float32
    expected_len = 12 * 12 + 2 + 1 + 4 + 1 + 1  # grid + pos + dir + inv + bot + tt
    assert flat.shape[0] == expected_len, f"flat len {flat.shape[0]} != {expected_len}"

    e.close()
    print(f"  Flat obs: OK (length={expected_len})")


def test_random_episodes() -> None:
    """Run 10 episodes with a random policy and report stats."""
    e = gymnasium.make("ThinkPuzzle-v0")

    total_rewards = []
    total_steps = []
    solves = 0
    n_episodes = 10

    for ep in range(n_episodes):
        obs, info = e.reset(seed=ep * 100)
        ep_reward = 0.0
        done = False

        while not done:
            action = e.action_space.sample()
            obs, reward, terminated, truncated, info = e.step(action)
            ep_reward += reward
            done = terminated or truncated

        total_rewards.append(ep_reward)
        total_steps.append(info["steps_taken"])
        if info["solved"]:
            solves += 1

    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    solve_rate = solves / n_episodes

    print(f"  Random policy ({n_episodes} episodes):")
    print(f"    Avg reward:  {avg_reward:.1f}")
    print(f"    Avg steps:   {avg_steps:.1f}")
    print(f"    Solve rate:  {solve_rate:.0%} ({solves}/{n_episodes})")

    e.close()


def test_info_dict() -> None:
    """Verify the info dict contains all required keys."""
    e = gymnasium.make("ThinkPuzzle-v0")
    obs, info = e.reset(seed=42)

    required_keys = [
        "is_deadlock", "steps_taken", "boxes_placed",
        "keys_collected", "doors_opened", "solved", "optimal_steps",
    ]
    for key in required_keys:
        assert key in info, f"Missing info key: {key}"

    # Step and check again
    obs, reward, term, trunc, info = e.step(0)
    for key in required_keys:
        assert key in info, f"Missing info key after step: {key}"

    e.close()
    print("  Info dict: OK")


def test_reset_options() -> None:
    """Verify reset options for difficulty and level_seed."""
    e = gymnasium.make("ThinkPuzzle-v0")

    # Reset with different difficulty
    obs, info = e.reset(options={"difficulty": 3, "level_seed": 12345})
    assert obs["grid"].shape == (12, 12)
    print(f"  Reset options: OK (optimal={info['optimal_steps']})")

    # Reset with same seed should produce identical level
    obs2, info2 = e.reset(options={"difficulty": 3, "level_seed": 12345})
    assert np.array_equal(obs["grid"], obs2["grid"]), "Same seed should give same grid"
    print("  Deterministic reset: OK")

    e.close()


def test_rgb_array() -> None:
    """Verify rgb_array rendering returns a valid numpy array."""
    e = gymnasium.make("ThinkPuzzle-v0", render_mode="rgb_array")
    obs, _ = e.reset(seed=42)

    frame = e.render()
    assert frame is not None, "rgb_array should return a frame"
    assert frame.ndim == 3, f"Expected 3D array, got {frame.ndim}D"
    assert frame.shape[2] == 3, f"Expected RGB (3 channels), got {frame.shape[2]}"
    assert frame.dtype == np.uint8, f"Expected uint8, got {frame.dtype}"
    print(f"  RGB array: OK (shape={frame.shape})")

    e.close()


def main() -> None:
    print("=" * 50)
    print("GYMNASIUM ENVIRONMENT TESTS")
    print("=" * 50)

    print("\n1. Registration:")
    test_registration()

    print("\n2. Observation / action spaces:")
    test_spaces()

    print("\n3. Flat observation:")
    test_flat_obs()

    print("\n4. Info dict:")
    test_info_dict()

    print("\n5. Reset options:")
    test_reset_options()

    print("\n6. Random policy episodes:")
    test_random_episodes()

    print("\n7. RGB array rendering:")
    test_rgb_array()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
