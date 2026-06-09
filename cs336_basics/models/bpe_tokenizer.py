import re

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = {}
        self.special_tokens =  special_tokens
        if special_tokens:
            self.pattern = "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True))
        else:
            self.pattern = None
        
        for merge in merges:
            tail = self.merges.get(merge[0], set())
            tail.add(merge[1])
            self.merges[merge[0]] = tail
        self.encoder = {v: k for k, v in self.vocab.items()}
        self.buffer = b''

    def __encode(self, text: str) -> list[int]:
        bytes_list = [bytes([b]) for b in text.encode("utf-8")]
        tokens = []
        #print(f"words: {bytes_list}")
        tmp = b''
        for byte in bytes_list:
            if tmp == b'':
                tmp += byte
            else:
                tail = self.merges.get(tmp, None)
                if tail is not None and byte in tail:
                    tmp += byte
                    #print(f"tmp: {tmp}")
                else:
                    tokens.append(self.encoder[tmp])
                    tmp = byte
            
        if tmp:
            tokens.append(self.encoder[tmp])
        #print(f"encode tokens: {tokens}")
        return tokens

    def encode(self, text: str) -> list[int]:
        output = []
        curr = 0
        print(f"text: {text}")
        if self.pattern:
            for match in re.finditer(self.pattern, text):
                #print(f"match: {match.start()}, {match.end()}, {match.group()}")
                output += self.__encode(text[curr:match.start()])
                #print(f"special token: {match.group()}, {match.group().encode('utf-8')}")
                #print(f"encodedspecial token: {self.encoder[match.group().encode("utf-8")]}")
                output += [self.encoder[match.group().encode("utf-8")]]
                #print(f"tmp text: {text[curr:match.start()]}")
                #print(f"output: {output}")
                curr = match.end()
            output += self.__encode(text[curr:])
        else:
            output = self.__encode(text)
        #print(f"output: {output}")
        return output


    def decode(self, ids: list[int]) -> str:
        #print(f"ids: {ids}")
        output = b''
        for id in ids:
            self.buffer += self.vocab[id]
        try:
            output = self.buffer.decode("utf-8")
            print(f"output: {output}")
            self.buffer = b''
            return output
        except UnicodeDecodeError:
            return ""
        