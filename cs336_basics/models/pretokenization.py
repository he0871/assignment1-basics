import os
from typing import BinaryIO

import regex as re

def find_chunk_boundaries(
    file: BinaryIO,
    chunk_size: int,
    split_special_token: bytes,
    pre_tokenizer: str,
) -> list[tuple[int, int]]:
    """
    Return byte ranges [start, end).

    Logic:
    1. Split file by special token.
    2. Exclude special tokens from output chunks.
    3. Inside each segment, split by GPT-2-style pre-tokenizer.
    4. Pack complete pre-tokens into chunks up to chunk_size.
       If next word does not fit, start a new chunk.
    """
    assert isinstance(split_special_token, bytes)
    assert chunk_size > 0

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    text = file.read().decode("utf-8", errors="ignore")
    special = split_special_token.decode("utf-8")

    ranges: list[tuple[int, int]] = []

    byte_pos = 0

    for segment in text.split(special):
        segment_bytes = segment.encode("utf-8")
        segment_start = byte_pos
        segment_end = segment_start + len(segment_bytes)

        _add_pretoken_chunks(
            ranges=ranges,
            segment=segment,
            segment_start=segment_start,
            chunk_size=chunk_size,
            pre_tokenizer=pre_tokenizer,
        )

        # move past segment + special token
        byte_pos = segment_end + len(split_special_token)

    return ranges


def _add_pretoken_chunks(
    ranges: list[tuple[int, int]],
    segment: str,
    segment_start: int,
    chunk_size: int,
    pre_tokenizer: str,
) -> None:
    curr_start: int | None = None
    curr_end: int | None = None

    for match in re.finditer(pre_tokenizer, segment):
        token = match.group()

        token_start = segment_start + len(segment[: match.start()].encode("utf-8"))
        token_end = segment_start + len(segment[: match.end()].encode("utf-8"))
        token_size = token_end - token_start

        if curr_start is None:
            curr_start = token_start
            curr_end = token_end
            continue

        assert curr_end is not None

        # If next pre-token does not fit, close current chunk.
        if token_end - curr_start > chunk_size:
            ranges.append((curr_start, curr_end))
            curr_start = token_start
            curr_end = token_end
        else:
            curr_end = token_end

    if curr_start is not None and curr_end is not None:
        ranges.append((curr_start, curr_end))