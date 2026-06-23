import cs336_basics.models.pretokenization as pretokenization
import regex as re
import os

from collections import Counter
from multiprocessing import Pool

pre_tokenizer = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenize_worker(chunk: bytes, special_tokens: list[bytes]) -> {tuple:int}:
    #print(f"chunk: {chunk}")
    segments = chunk.split(b"<|endoftext|>")
    word_freq = Counter()
    for segement in segments:
        for word in re.findall(pre_tokenizer, segement):
            word_freq[tuple(bytes([b]) for b in word)] += 1
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

def __iter_chunks(input_path, boundaries, special_tokens_bytes):
    with open(input_path, "rb") as f:
        pt = 0
        for idx in boundaries:
            f.seek(pt)
            #print(f"pt: {pt}, idx: {idx}")
            chunk = f.read(idx - pt)
            if chunk:
                yield (chunk, special_tokens_bytes)
            pt = idx

def __pre_tokenize_worker_star(args):
    #print(f"args: {args}")
    chunk, special_tokens_bytes = args
    return pre_tokenize_worker(chunk, special_tokens_bytes)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    initial_vocab_size = 256 + len(special_tokens)
    num_merges = vocab_size - initial_vocab_size
    special_tokens_bytes = [token.encode("utf-8") for token in special_tokens]
    word_freqs = Counter()
    chunk_szie = 409600

    total_counter = Counter()
    with open(input_path, "rb") as f:
        boundaries = pretokenization.find_chunk_boundaries(f, chunk_szie, b"<|endoftext|>")
 
    with Pool() as pool:
        counters = pool.imap_unordered(
            __pre_tokenize_worker_star,
            __iter_chunks(input_path, boundaries, special_tokens_bytes),
            chunksize=2,
        )
        for counter in counters:
            total_counter.update(counter)

    print(f"total_counter len: {len(total_counter)}")
    word_freqs.update(total_counter)

    pair_counts = get_pair_counts(word_freqs)
    #print(f"pair_counts: {pair_counts}")

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
        #print(f"best_pair: {best_pair}")

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        word_freqs = apply_merge(word_freqs, best_pair)

    return vocab, merges
         
if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab, merges = train_bpe(input_path, 500, ["<|endoftext|>"]) 
   #print(len(vocab))
   #print(len(merges))