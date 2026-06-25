if __name__ == "__main__":
    base_path = "/Users/jingyuanhe/code/assignment1-basics"
    train_name = "owt_train"
    train_path = f"{base_path}/data/{train_name}.txt"
    save_path = f"{base_path}/data/{train_name}_subset.txt"

    special_token = "<|endoftext|>"
    chunk_size = 2000000
    first_10_docs = []
    with open(train_path, "r", encoding="utf-8") as f:
    
        text = str(f.read(chunk_size))
        if special_token in text:
            docs = text.split(special_token)
            print(f"found {len(docs)} docs")
            first_10_docs.extend(docs[:10])
     



    with open(save_path, "w", encoding="utf-8") as f:
        f.write(special_token.join(first_10_docs))
