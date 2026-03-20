"""Unit tests for token-based HiCache L3 existence queries."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

import unittest

from sglang.srt.mem_cache.hicache_query_utils import query_hicache_exists_by_tokens
from sglang.srt.mem_cache.hicache_storage import get_hash_str


class FakeStorageBackend:
    def __init__(self, existing_hashes):
        self.existing_hashes = set(existing_hashes)
        self.calls = []

    def batch_exists(self, keys, extra_info=None):
        self.calls.append((list(keys), extra_info))
        for i, key in enumerate(keys):
            if key not in self.existing_hashes:
                return i
        return len(keys)


class TestHiCacheQueryUtils(unittest.TestCase):
    def test_exists_by_tokens_returns_per_page_results(self):
        page_size = 4
        token_ids = list(range(1, 11))
        hash_0 = get_hash_str(token_ids[:4])
        hash_1 = get_hash_str(token_ids[4:8], prior_hash=hash_0)
        backend = FakeStorageBackend({hash_0})

        result = query_hicache_exists_by_tokens(
            token_ids=token_ids,
            page_size=page_size,
            storage_backend=backend,
        )

        self.assertEqual(result.page_size, page_size)
        self.assertEqual(result.input_token_count, 10)
        self.assertEqual(result.aligned_token_count, 8)
        self.assertEqual(result.page_hashes, [hash_0, hash_1])
        self.assertEqual(result.exists, [True, False])
        self.assertEqual(result.longest_prefix_pages, 1)
        self.assertEqual(result.longest_prefix_tokens, 4)
        self.assertEqual(backend.calls, [([hash_0], None), ([hash_1], None)])

    def test_exists_by_tokens_handles_short_prefix(self):
        backend = FakeStorageBackend(set())

        result = query_hicache_exists_by_tokens(
            token_ids=[1, 2, 3],
            page_size=4,
            storage_backend=backend,
        )

        self.assertEqual(result.input_token_count, 3)
        self.assertEqual(result.aligned_token_count, 0)
        self.assertEqual(result.page_hashes, [])
        self.assertEqual(result.exists, [])
        self.assertEqual(result.longest_prefix_pages, 0)
        self.assertEqual(result.longest_prefix_tokens, 0)
        self.assertEqual(backend.calls, [])

    def test_exists_by_tokens_supports_eagle_bigram_keys(self):
        token_ids = [10, 11, 12, 13, 14]
        page_0 = [(10, 11), (11, 12)]
        page_1 = [(12, 13), (13, 14)]
        hash_0 = get_hash_str(page_0)
        hash_1 = get_hash_str(page_1, prior_hash=hash_0)
        backend = FakeStorageBackend({hash_0, hash_1})

        result = query_hicache_exists_by_tokens(
            token_ids=token_ids,
            page_size=2,
            storage_backend=backend,
            is_eagle=True,
        )

        self.assertEqual(result.input_token_count, 5)
        self.assertEqual(result.aligned_token_count, 4)
        self.assertEqual(result.page_hashes, [hash_0, hash_1])
        self.assertEqual(result.exists, [True, True])
        self.assertEqual(result.longest_prefix_pages, 2)
        self.assertEqual(result.longest_prefix_tokens, 4)


if __name__ == "__main__":
    unittest.main()
