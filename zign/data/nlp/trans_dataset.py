from tqdm import tqdm
from zign.data.dataset import zDataset
from zign.data.nlp.text import Text


class TransDataset(zDataset):
    
    def __init__(self, src: Text, tgt: Text):
        self.src = src
        self.tgt = tgt
        
        self.data = []
        if len(self.tgt) == 0:
            self.tgt = ['' for i in self.src]
        for (src, tgt) in tqdm(zip(self.src.tokenize(), self.tgt.tokenize()), ncols=80):
            self.data.append((src, tgt))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]