import unittest

import numpy as np

from src import collect


def demo_data():
    lengths = np.array([2, 3, 1], dtype=np.int64)
    starts = np.array([0, 2, 5], dtype=np.int64)
    observations = np.arange(12, dtype=np.float32).reshape(6, 2)
    actions = np.arange(6, dtype=np.float32).reshape(6, 1)
    return {
        "observations": observations,
        "actions": actions,
        "episode_returns": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "episode_lengths": lengths,
        "episode_starts": starts,
    }


class SubsetTests(unittest.TestCase):
    def test_subset_keeps_whole_episode_boundaries(self):
        obs, acts = collect.subset(demo_data(), 2)

        np.testing.assert_array_equal(obs, demo_data()["observations"][:5])
        np.testing.assert_array_equal(acts, demo_data()["actions"][:5])

    def test_subset_rejects_non_positive_episode_count(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            collect.subset(demo_data(), 0)

    def test_subset_rejects_more_episodes_than_available(self):
        with self.assertRaisesRegex(ValueError, "requested 4 episodes"):
            collect.subset(demo_data(), 4)

    def test_subset_rejects_inconsistent_episode_starts(self):
        data = demo_data()
        data["episode_starts"] = np.array([0, 3, 5], dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "episode_starts"):
            collect.subset(data, 2)

    def test_subset_rejects_mismatched_observation_action_counts(self):
        data = demo_data()
        data["actions"] = data["actions"][:-1]

        with self.assertRaisesRegex(ValueError, "observations and actions"):
            collect.subset(data, 2)


if __name__ == "__main__":
    unittest.main()
