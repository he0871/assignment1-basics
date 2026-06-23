# Run single test
```bash
uv run pytest tests/test_model.py::test_linear -vv
```

uv run pytest tests/test_tokenizer.py -q --no-header -rf



current progress
```bash
uv run pytest tests/test_tokenizer.py::test_ascii_string_matches_tiktoken  -vv
```

```bash
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   tests/adapters.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        cs336_basics/models/
        data/
        note.md

```


Test Out
```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
print(enc.encode("a\n\n<|endoftext|>b", allowed_special={"<|endoftext|>"}))
print(enc.encode("a\n\n", allowed_special={"<|endoftext|>"}))
```


# py-spy usage
```bash
# Live bottleneck view
py-spy top -- uv run pytest tests/test_tokenizer.py -vv

# Profile one script
py-spy record -o profile.svg -- python main.py

# Attach to already running process
py-spy top --pid <PID>
py-spy record -o profile.svg --pid <PID>

# If code uses subprocesses
py-spy record --subprocesses -o profile.svg -- uv run pytest tests/test_tokenizer.py
```

sudo py-spy record --subprocesses -o profile.svg -- python cs336_basics/experiments/train_10k_tinyStories.py

# Optimizer

## before optimizer word_frequcy counter 

| Hotspot | ~% of total samples | What it is doing |
|---|---|---|
| `train_bpe` line 106 | ~15% | `get_pair_counts(word_freqs)` inside the merge loop |
| `train_bpe` line 117 | ~14% | `word_freqs = apply_merge(...)` |
| `get_pair_counts` lines 24–25 | ~14% combined | Inner loops over every word and every adjacent pair |
| `apply_merge` lines 47–48 | ~13% combined | Rebuilding the full word-frequency dict each merge |
| `merge_word` line 34 | ~4% | Scanning each word to apply one merge |
| `pre_tokenize_worker` (workers) | ~20% total across workers | Pretokenization + regex; runs in parallel |

31 minutes 4 seconds

```python
    while len(merges) < num_merges:
        pair_counts = get_pair_counts(word_freqs)
        

        if not pair_counts:
            break

        best_pair =  max( pair_counts.items(),key=lambda item: (item[1], item[0]))[0]
        #print(f"best_pair: {best_pair}")

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        word_freqs = apply_merge(word_freqs, best_pair)
```