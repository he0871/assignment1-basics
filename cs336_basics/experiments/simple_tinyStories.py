from cs336_basics.models.bpe_tokenizer import BPETokenizer


if __name__ == "__main__":
    vocab_path = "cs336_basics/tokenizers/TinyStoriesV2-GPT4-train-10k.vocab"
    merges_path = "cs336_basics/tokenizers/TinyStoriesV2-GPT4-train-10k.merges"
    tokenizer = BPETokenizer({}, [], [])
    tokenizer.from_files(vocab_path, merges_path)
    with open("data/TinyStoriesV2-GPT4-train_subset.txt", "rb") as f:
        text = f.read()
    ids = tokenizer.encode(str(text))
    print(f"ids length: {len(ids)}, text length: {len(text)}, compression ratio: {len(ids) / len(text)}")
