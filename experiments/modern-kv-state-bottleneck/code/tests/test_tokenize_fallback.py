import sys
import types
import unittest

from sglang_hicache.workload import checkpoint_tokenize_fallback


class TokenizeFallbackTests(unittest.TestCase):
    def test_primary_tokenizer_is_preferred(self):
        tokenize = checkpoint_tokenize_fallback(lambda text: [1, len(text)], "/model")
        self.assertEqual(tokenize("abc"), [1, 3])

    def test_same_checkpoint_fallback_is_lazy_and_reused(self):
        calls = []

        class FakeTokenizer:
            def encode(self, text, add_special_tokens):
                if add_special_tokens:
                    raise AssertionError("special tokens must stay disabled")
                return [7, len(text)]

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path, local_files_only, trust_remote_code):
                calls.append((path, local_files_only, trust_remote_code))
                return FakeTokenizer()

        old = sys.modules.get("transformers")
        sys.modules["transformers"] = types.SimpleNamespace(
            AutoTokenizer=FakeAutoTokenizer
        )
        try:
            def fail(_text):
                raise RuntimeError("tokenize response cannot be serialized")

            tokenize = checkpoint_tokenize_fallback(fail, "/models/gemma")
            self.assertEqual(tokenize("abcd"), [7, 4])
            self.assertEqual(tokenize("xy"), [7, 2])
            self.assertEqual(calls, [("/models/gemma", True, True)])
        finally:
            if old is None:
                del sys.modules["transformers"]
            else:
                sys.modules["transformers"] = old


if __name__ == "__main__":
    unittest.main()
