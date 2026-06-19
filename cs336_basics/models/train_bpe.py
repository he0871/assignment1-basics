import cs336_basics.models.pretokenization as pretokenization
import regex as re
import os

from collections import Counter
from multiprocessing import Pool

pre_tokenizer = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenize_worker(chunk: str, special_tokens: list[str]) -> {tuple:int}:
    word_freq = Counter()
    for match in re.finditer(pre_tokenizer, chunk):
        word = match.group()
        if word not in special_tokens:
            word_freq[tuple(bytes([b]) for b in word.encode("utf-8"))] += 1
    return word_freq

def get_pair_counts(word_freqs):
    pair_counts = Counter()

    for word, freq in word_freqs.items():
        for a, b in zip(word, word[1:]):
            pair_counts[(a, b)] += freq

    return pair_counts

def merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged = []
    i = 0

    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1

    return tuple(merged)

def apply_merge(word_freq, pair):
    new_word_freq = Counter()

    for word, freq in word_freq.items():
        new_word = merge_word(word, pair)
        new_word_freq[new_word] += freq

    return new_word_freq

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    initial_vocab_size = 256 + len(special_tokens)
    num_merges = vocab_size - initial_vocab_size
    word_freqs = Counter()
    chunk_size = 1024
    tasks = []
    with open(input_path, "rb") as f:
        ranges = pretokenization.find_chunk_boundaries(f, chunk_size, b"<|endoftext|>", pre_tokenizer)
         
        for start, end in ranges:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            tasks.append((chunk, special_tokens))
    
    print(f"tasks len: {len(tasks)}")
    with Pool() as pool:
        counters = pool.starmap(pre_tokenize_worker, tasks)

    for counter in counters:
        word_freqs.update(counter)

    pair_counts = get_pair_counts(word_freqs)

    vocab: dict[int, bytes] = {
        i: bytes([i])
        for i in range(0,256)
    }
    vocab[256] = b"<|endoftext|>"
    merges = []
    while len(merges) < num_merges:
        pair_counts = get_pair_counts(word_freqs)

        if not pair_counts:
            break

        best_pair =  max( pair_counts.items(),key=lambda item: (item[1], item[0]))[0]

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        word_freqs = apply_merge(word_freqs, best_pair)

    return vocab, merges
         
if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab, merges = train_bpe(input_path, 500, ["<|endoftext|>"]) 
   #print(len(vocab))
   #print(len(merges))