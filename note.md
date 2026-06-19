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

py-spy record --subprocesses -o profile.svg -- uv run 