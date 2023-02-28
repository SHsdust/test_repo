import torch
import torch.nn as nn
import logging
from transformers import AutoModel
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)


class BertRNN(nn.Module):
    def __init__(self, nlayer, nclass, emb_batch, dropout=0.5, nfinetune=0):
        super(BertRNN, self).__init__()
        self.emb_batch = emb_batch
        self.bert = AutoModel.from_pretrained('roberta-base')
        nhid = self.bert.config.hidden_size

        for param in self.bert.parameters():
            param.requires_grad = False
        n_layers = 12
        if nfinetune > 0:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            for i in range(n_layers - 1, n_layers - 1 - nfinetune, -1):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

        # classifying act tag
        self.encoder = nn.GRU(nhid, nhid // 2, num_layers=nlayer, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(nhid, nclass)

        self.dropout = nn.Dropout(p=dropout)
        self.nclass = nclass
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)


    def forward(self, input_ids, attention_mask, speaker_ids, chunk_lens, utterance_attention_mask, rm_labels, w_da, w_rr, w_rm, da_labels=None):
        '''
        da_labels: train阶段[B, chunk_size], inference阶段为None
        '''
        chunk_lens = chunk_lens.to('cpu')
        batch_size, chunk_size, seq_len = input_ids.shape
        speaker_ids = speaker_ids.reshape(-1)   # [B, chunk_size]->[B*chunk_size]

        input_ids = input_ids.reshape(-1, seq_len)  # [B, chunk_size, L]->[B*chunk_size, L]
        attention_mask = attention_mask.reshape(-1, seq_len)  # [B, chunk_size, L]->[B*chunk_size, L]

        # embeddings = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B*chunk_size, E]
        if self.emb_batch == 0:
            embeddings = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]   # [B*chunk_size, E]
        else:
            embeddings_ = []
            dataset2 = TensorDataset(input_ids, attention_mask)
            loader = DataLoader(dataset2, batch_size=self.emb_batch)
            for _, batch in enumerate(loader):
                embeddings = self.bert(batch[0], attention_mask=batch[1]).last_hidden_state[:, 0, :]   # [emb_batch, E]
                embeddings_.append(embeddings)
                embeddings = torch.cat(embeddings_, dim=0)

        nhid = embeddings.shape[-1]

        embeddings = embeddings.reshape(-1, chunk_size, nhid)  # (bs, chunk_size, emd_dim)
        embeddings = embeddings.permute(1, 0, 2)  # (chunk_size, bs, emb_dim)

        outputs, _ = self.encoder(embeddings) #[chunk_size, B, E]
        outputs = outputs.transpose(0, 1)  # [B, chunk_size, E]

        outputs = self.dropout(outputs)

        da_outputs = self.fc(outputs)  # [B, chunk_size, emb_dim]


        da_outputs = da_outputs.reshape(-1, self.nclass)  # (bs*chunk_size, nclass)


        if da_labels is None:
            ce_loss = None
        else:
            da_labels = da_labels.reshape(-1)  # [B*chunk_size]
            ce_loss = self.loss_fn(da_outputs, da_labels)


        total_loss = w_da * ce_loss
        return da_outputs, total_loss
