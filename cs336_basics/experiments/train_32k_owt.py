if __name__ == "__main__":


    import cs336_basics.models.train_bpe   as train_bpe
    import cs336_basics.models.bpe_tokenizer as bpe_tokenizer
    import json

    base_path = "/Users/jingyuanhe/code/assignment1-basics"

    train_name = "owt_train"
    train_path = f"{base_path}/data/{train_name}.txt"
    vocab_path = f"{base_path}/cs336_basics/tokenizers/{train_name}-32k.vocab"
    merges_path = f"{base_path}/cs336_basics/tokenizers/{train_name}-32k.merges"

    special_token = "<|endoftext|>"



    vocab, merges = train_bpe.train_bpe(train_path, 32000, [special_token])
    #print(f"vocab: {vocab}")
    with open(vocab_path, "w") as f:
        f.write(str(vocab))
    with open(merges_path, "w") as f:
        f.write(str(merges))
    #bpe_tokenizer.from_files(vocab_path, merges_path)