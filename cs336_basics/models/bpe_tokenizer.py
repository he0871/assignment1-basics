import regex as re

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = {}
        for merge in merges:
            sub_rank = self.merges.get(merge[0], [])
            sub_rank.append(merge[1])
            self.merges[merge[0]] = sub_rank
        self.special_tokens =  special_tokens
        if special_tokens:
            self.pattern = "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True))
        else:
            self.pattern = None
        self.pre_tokenizer = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        self.encoder = {v: k for k, v in self.vocab.items()}
        self.buffer = b''
        self.format_ch = {b'\n'}
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(merges)
        }
        #print(f"merge_ranks: {self.merge_ranks}")
        #self.merges = merges

    def __merge(self, bytes_list: list[bytes]) -> list[bytes]:
        while True:
            pairs = [
            (self.merge_ranks[(bytes_list[i], bytes_list[i + 1])], i)
            for i in range(len(bytes_list) - 1)
            if (bytes_list[i], bytes_list[i + 1]) in self.merge_ranks
            ]
            if not pairs:
                break
            _, merge_idx = min(pairs)
            bytes_list = (
                bytes_list[:merge_idx]
                + [bytes_list[merge_idx] + bytes_list[merge_idx + 1]]
                + bytes_list[merge_idx + 2:]
            )

        return bytes_list

    def __encode(self, text: str, pos:int) -> list[int]:
        bytes_list = [bytes([b]) for b in text.encode("utf-8")]
        output = []
        # skip format characters at the beginning
        if pos == 0:
            i = 0
            while i < len(bytes_list):
                if bytes_list[i] in self.format_ch:
                    output.append(self.encoder[bytes_list[i]])
                    i += 1
                else:
                    break
            bytes_list = bytes_list[i:]
        # merge bytes_list
        while True:
            merged = self.__merge(bytes_list)
            if len(merged) == len(bytes_list):
                break
            bytes_list = merged
        #print(f"input: {text}, output: {output + [self.encoder[token] for token in bytes_list]}")
        return output + [self.encoder[token] for token in bytes_list]
                

    def encode(self, text: str) -> list[int]:
        output = []
        curr = 0
        #print(f"text: {text}")
        pos = 0
        if self.pattern:
            # split text by special tokens
            for match in re.finditer(self.pattern, text):
                #print(f"match: {match.start()}, {match.end()}, {match.group()}")
                pre_token = text[curr:match.start()]
                # pre-tokenize the text
                idx = 0
                for pt_match in re.finditer(self.pre_tokenizer, pre_token):
                    #print(f"pt_match: {pt_match.start()}, {pt_match.end()}, {pt_match.group()}")
                    #output += self.__encode(pre_token[idx:pt_match.start()], pos)
                    output += self.__encode(pt_match.group(), pos)
                    #print(f"output: {output}")
                    if len(output) > 0:
                        pos += 1
                    idx = pt_match.end()
                special_token = match.group()
                output += [self.encoder[special_token.encode("utf-8")]]
                curr = match.end()
                #pos += 1
            output += self.__encode(text[curr:], pos)
        else:
            output = self.__encode(text, pos)
        
        #print(f"output: {output}")
        return output


    def decode(self, ids: list[int]) -> str:
        #print(f"ids: {ids}")
        output = b''
        for id in ids:
            self.buffer += self.vocab[id]
        try:
            output = self.buffer.decode("utf-8")
            #print(f"output: {output}")
            self.buffer = b''
            return output
        except UnicodeDecodeError:
            return ""
        