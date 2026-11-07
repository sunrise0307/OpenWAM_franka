import os
from unittest import TestCase
from unittest.mock import patch

from rot6d_deploy.runtime import BUILD_DIR, PYTHON_DIR, bootstrap, build_environment


class RuntimeTests(TestCase):
    def test_build_environment_puts_one_build_first(self):
        environment = build_environment({
            "POLYMETIS_BUILD_DIR": "/old/build",
            "PYTHONPATH": "/old/python",
            "LD_LIBRARY_PATH": "/old/torch_isolation",
        })
        self.assertEqual(environment["POLYMETIS_BUILD_DIR"], str(BUILD_DIR))
        self.assertEqual(environment["PYTHONPATH"].split(os.pathsep)[0], str(PYTHON_DIR))
        self.assertEqual(environment["LD_LIBRARY_PATH"].split(os.pathsep)[:2], [
            str(BUILD_DIR / "torch_isolation/pinocchio_isolation"),
            str(BUILD_DIR / "torch_isolation"),
        ])

    def test_bootstrap_restarts_before_imports_when_build_is_missing(self):
        with patch.dict(os.environ, {"POLYMETIS_BUILD_DIR": "/old/build"}), \
                patch("os.execvpe") as restart:
            bootstrap()
        self.assertEqual(restart.call_count, 1)
        self.assertEqual(restart.call_args.args[2]["POLYMETIS_BUILD_DIR"], str(BUILD_DIR))


if __name__ == "__main__":
    import unittest
    unittest.main()
