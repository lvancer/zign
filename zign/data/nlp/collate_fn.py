import torch
from torch.nn.utils.rnn import pad_sequence
from zign.data.nlp.vocab import Vocab


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