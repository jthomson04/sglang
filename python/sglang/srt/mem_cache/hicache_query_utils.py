from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from sglang.srt.mem_cache.hicache_storage import get_hash_str
from sglang.srt.mem_cache.radix_cache import page_align_keys
from sglang.srt.mem_cache.utils import convert_to_bigram_key


@dataclass
class HiCacheExistsByTokensResult:
    page_size: int
    input_token_count: int
    aligned_token_count: int
    page_hashes: List[str]
    exists: List[bool]
    longest_prefix_pages: int
    longest_prefix_tokens: int


def query_hicache_exists_by_tokens(
    *,
    token_ids: List[int],
    page_size: int,
    storage_backend: Any,
    is_eagle: bool = False,
    extra_key: Optional[str] = None,
) -> HiCacheExistsByTokensResult:
    """Compute logical HiCache page hashes and check L3 existence per page.

    `extra_key` is accepted for API stability, but is currently not part of the
    L3 hash construction in HiCache.
    """
    del extra_key

    input_token_count = len(token_ids)
    query_tokens = convert_to_bigram_key(token_ids) if is_eagle else list(token_ids)
    aligned_tokens = page_align_keys(query_tokens, page_size)

    page_hashes: List[str] = []
    exists: List[bool] = []
    last_hash = None

    for start in range(0, len(aligned_tokens), page_size):
        page_tokens = aligned_tokens[start : start + page_size]
        last_hash = get_hash_str(page_tokens, prior_hash=last_hash)
        page_hashes.append(last_hash)
        exists.append(storage_backend.batch_exists([last_hash]) == 1)

    longest_prefix_pages = 0
    for page_exists in exists:
        if not page_exists:
            break
        longest_prefix_pages += 1

    return HiCacheExistsByTokensResult(
        page_size=page_size,
        input_token_count=input_token_count,
        aligned_token_count=len(aligned_tokens),
        page_hashes=page_hashes,
        exists=exists,
        longest_prefix_pages=longest_prefix_pages,
        longest_prefix_tokens=longest_prefix_pages * page_size,
    )
