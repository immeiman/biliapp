import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

import config


class DotenvConfigTests(unittest.TestCase):
    def test_load_dotenv_sets_missing_values(self):
        key_plain = "BILIRUBIN_TEST_DOTENV_PLAIN"
        key_quoted = "BILIRUBIN_TEST_DOTENV_QUOTED"
        key_exported = "BILIRUBIN_TEST_DOTENV_EXPORTED"

        with tempfile.TemporaryDirectory() as tmpdir:
            dotenv = Path(tmpdir) / ".env"
            dotenv.write_text(
                "\n".join(
                    [
                        "# local test configuration",
                        f"{key_plain}=123 # inline comment",
                        f'{key_quoted}="hello # still value"',
                        f"export {key_exported}=yes",
                        "1INVALID_KEY=ignored",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(key_plain, None)
                os.environ.pop(key_quoted, None)
                os.environ.pop(key_exported, None)

                self.assertTrue(config._load_dotenv(dotenv))
                self.assertEqual(os.environ[key_plain], "123")
                self.assertEqual(os.environ[key_quoted], "hello # still value")
                self.assertEqual(os.environ[key_exported], "yes")
                self.assertNotIn("1INVALID_KEY", os.environ)

    def test_load_dotenv_does_not_override_existing_environment(self):
        key = "BILIRUBIN_TEST_DOTENV_KEEP"

        with tempfile.TemporaryDirectory() as tmpdir:
            dotenv = Path(tmpdir) / ".env"
            dotenv.write_text(f"{key}=from-dotenv\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {key: "from-shell"}):
                self.assertTrue(config._load_dotenv(dotenv))
                self.assertEqual(os.environ[key], "from-shell")

    def test_load_dotenv_returns_false_when_file_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(config._load_dotenv(Path(tmpdir) / ".env"))


if __name__ == "__main__":
    unittest.main()
