from collections import Counter
from torchtext.vocab import vocab
from torchtext.vocab import Vocab as TorchTextVocab 
from tqdm import tqdm

from torchtext.data.utils import get_tokenizer

class Vocab(TorchTextVocab):
    
    def __init__(self, tokenizer, sentences, min_freq=1, specials=['<unk>', '<pad>', '<bos>', '<eos>']):
        self.tokenizer = tokenizer 
        self.specials = specials # [未知，填充，开始，结束]
        counter = Counter()
        for sentence in tqdm(sentences):
            counter.update(tokenizer(sentence))
        super().__init__(vocab(counter, specials=specials, min_freq=min_freq).vocab)
        
        self.UNK_IDX = self[specials[0]]
        self.PAD_IDX = self[specials[1]]
        self.BOS_IDX = self[specials[2]]
        self.EOS_IDX = self[specials[3]]
        
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_specials(self):
        return self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX
    
    def get_UNK(self):
        return (self.specials[0], self.UNK_IDX)
    
    def get_PAD(self):
        return (self.specials[1], self.PAD_IDX)
    
    def get_BOS(self):
        return (self.specials[2], self.BOS_IDX)
    
    def get_EOS(self):
        return (self.specials[3], self.EOS_IDX)