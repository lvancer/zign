from collections import Counter
from torchtext.vocab import vocab
from torchtext.vocab import Vocab as TorchTextVocab 
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from torch.utils import data
from torchtext.datasets import IWSLT2016

from config import Config
from zign.data import zDataset
from zign.utils import io
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


class Dataset(zDataset):
    
    def __init__(self, src: Text, tgt: Text):
        self.src = src
        self.tgt = tgt
        
        self.data = []
        if len(self.tgt) == 0:
            self.tgt = ['' for i in self.src]
        for (src, tgt) in tqdm(zip(self.src.tokenize(), self.tgt.tokenize()), ncols=80):
            self.data.append((src, tgt))
        
    def get_vocab(self):
        return self.src.vocab, self.tgt.vocab
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


def pad_sequence_batch(vocab: Vocab):
    def _pad_sequence_batch(data_batch):
        """
        自定义一个函数来对每个batch的样本进行处理，该函数将作为一个参数传入到类DataLoader中。
        由于在DataLoader中是对每一个batch的数据进行处理，所以这就意味着下面的pad_sequence操作，最终表现出来的结果就是
        不同的样本，padding后在同一个batch中长度是一样的，而在不同的batch之间可能是不一样的。因为pad_sequence是以一个batch中最长的
        样本为标准对其它样本进行padding
        :param data_batch:
        :return:
        """
        src_batch, tgt_batch = [], []
        for (src_item, tgt_item) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            src_batch.append(src_item)  # 编码器输入序列不需要加起止符
            # 在每个idx序列的首位加上 起始token 和 结束 tself.vocab.get_stoi().get(token, self.vocab.UNK_IDX)oken
            tgt = torch.cat([torch.tensor([vocab.BOS_IDX]), tgt_item, torch.tensor([vocab.EOS_IDX])], dim=0)
            tgt_batch.append(tgt)
        # 以最长的序列为标准进行填充
        src_batch = pad_sequence(src_batch, padding_value=vocab.PAD_IDX)  # [src_len, batch_size]
        tgt_batch = pad_sequence(tgt_batch, padding_value=vocab.PAD_IDX)  # [tgt_len, batch_size]
        return src_batch, tgt_batch
    return _pad_sequence_batch
    

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(ignore_index, src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)  # [tgt_len,tgt_len]
    # Decoder的注意力Mask输入，用于掩盖当前position之后的position，所以这里是一个对称矩阵

    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    # Encoder的注意力Mask输入，这部分其实对于Encoder来说是没有用的，所以这里全是0

    src_padding_mask = (src == ignore_index).transpose(0, 1)
    # False表示not masked, True表示masked
    # 用于mask掉Encoder的Token序列中的padding部分,[batch_size, src_len]
    tgt_padding_mask = (tgt == ignore_index).transpose(0, 1)
    # 用于mask掉Decoder的Token序列中的padding部分,batch_size, tgt_len
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask         



         