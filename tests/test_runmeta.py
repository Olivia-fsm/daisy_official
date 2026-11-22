"""runmeta.append_line / retry_io retry a failed write (a transient file-system error) and raise after the last
retry."""

import builtins
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from daisy.runmeta import append_line  # noqa: E402


class AppendLineTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "log.jsonl")

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_retry_then_success(self):
        real, calls = builtins.open, {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise PermissionError(13, "Permission denied")
            return real(*a, **k)

        append_line(self.path, "a")
        with mock.patch("builtins.open", flaky), mock.patch("daisy.runmeta.time.sleep") as sl:
            append_line(self.path, "b")
        self.assertEqual(sl.call_count, 1)
        with open(self.path) as f:
            self.assertEqual(f.read(), "a\nb\n")

    def test_persistent_failure_raises(self):
        with mock.patch("builtins.open", mock.Mock(side_effect=PermissionError(13, "Permission denied"))), \
                mock.patch("daisy.runmeta.time.sleep"):
            with self.assertRaises(PermissionError):
                append_line(self.path, "x")


if __name__ == "__main__":
    unittest.main()
