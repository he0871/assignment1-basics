import os
from typing import BinaryIO
import regex as re

def find_chunk_boundaries(
    file: BinaryIO,
    chunk_size: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    PAT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_boundaries = [0, file_size]
    current_position = 0

    while True:
        chunk = file.read(chunk_size)
        if chunk == b"":
            break
        found_at = chunk.find(split_special_token)
        if found_at != -1:
            chunk_boundaries.insert(-1, current_position + found_at)
        else:
            matches = list(re.finditer(PAT, chunk))
            if matches and len(matches) > 1:
                chunk_boundaries.insert(-1, current_position + matches[-2].end())

        current_position += len(chunk) 
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))