from cs336_basics.models.bpe_tokenizer import BPETokenizer
import time
import numpy as np

if __name__ == "__main__":
    start_time = time.time()
    vocab_path = "cs336_basics/tokenizers/owt_train-32k.vocab"
    merges_path = "cs336_basics/tokenizers/owt_train-32k.merges"
    special_token = "<|endoftext|>"
    tokenizer = BPETokenizer({}, [], [special_token])
    tokenizer.from_files(vocab_path, merges_path)
    chunk_size = 100000000 
    cap = 0
    cnt = 0
    with open("data/owt_train.txt", "r") as f, open("data/owt_train_encoded.txt", "ab") as out_file:
        data = f.read(chunk_size)
        while len(data) > 0:
            ids = tokenizer.encode(data)
            ids = np.array(ids, dtype=np.uint16)
            out_file.write(ids)
            data = f.read(chunk_size)
            cnt += len(data)
            if cap > 0 and cnt > cap:
                break
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds, total bytes: {cnt}")
        