if __name__ == "__main__":
    base_path = "/Users/jingyuanhe/code/assignment1-basics"
    train_name = "TinyStoriesV2-GPT4-train"
    train_path = f"{base_path}/data/{train_name}.txt"
    save_path = f"{base_path}/data/{train_name}_subset.txt"

    special_token = "<|endoftext|>"

    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()

    docs = text.split(special_token)

    first_10_docs = docs[:10]

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(special_token.join(first_10_docs))
