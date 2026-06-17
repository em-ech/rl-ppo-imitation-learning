import unittest


class ImportTests(unittest.TestCase):
    def test_lightweight_package_imports_do_not_require_rl_stack(self):
        import src
        from src import collect, config, seeding

        self.assertIn("collect", src.__all__)
        self.assertIn("Walker2d-v4", config.ENV_IDS)
        self.assertTrue(callable(seeding.set_seed))
        self.assertTrue(callable(collect.subset))


if __name__ == "__main__":
    unittest.main()
