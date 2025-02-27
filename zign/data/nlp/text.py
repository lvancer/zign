import torch
from tqdm import tqdm

from zign.data.nlp.vocab import Vocab

class Text:
    
    def __init__(self, sentences, vocab: Vocab):
        self.sentences = sentences
        self.vocab = vocab
        
    def __len__(self):
        return len(self.sentences)
        
    def tokenize(self):
        """
        将每一句话中的每一个词根据字典转换成索引的形式
        """
        for raw in tqdm(self.sentences):
            # ids = []
            # for token in self.vocab.tokenizer(raw.rstrip("\n")):
            #     if token in self.vocab:
            #         ids.append(self.vocab[token])
            #     else:
            #         ids.append(self.vocab.UNK_IDX)
            # yield torch.tensor(ids, dtype=torch.long)
            yield torch.tensor([(self.vocab[token] if token in self.vocab else self.vocab.UNK_IDX) for token in self.vocab.tokenizer(raw.rstrip("\n"))], dtype=torch.long)
