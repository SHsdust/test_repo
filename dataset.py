import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import logging
from transformers import AutoTokenizer
from utils import swda_label2id, swda_train_set_idx, swda_valid_set_idx, swda_test_set_idx
import re

logger = logging.getLogger(__name__)
# ## Dialogue act label encoding, SWDA
# {'qw^d': 0, '^2': 1, 'b^m': 2, 'qy^d': 3, '^h': 4, 'bk': 5, 'b': 6, 'fa': 7, 'sd': 8, 'fo_o_fw_"_by_bc': 9,
#              'ad': 10, 'ba': 11, 'ng': 12, 't1': 13, 'bd': 14, 'qh': 15, 'br': 16, 'qo': 17, 'nn': 18, 'arp_nd': 19,
#              'fp': 20, 'aap_am': 21, 'oo_co_cc': 22, 'h': 23, 'qrr': 24, 'na': 25, 'x': 26, 'bh': 27, 'fc': 28,
#              'aa': 29, 't3': 30, 'no': 31, '%': 32, '^g': 33, 'qy': 34, 'sv': 35, 'ft': 36, '^q': 37, 'bf': 38,
#              'qw': 39, 'ny': 40, 'ar': 41, '+': 42}

# ## Topic label encoding, SWDA
# {'CARE OF THE ELDERLY': 0, 'HOBBIES AND CRAFTS': 1, 'WEATHER CLIMATE': 2, 'PETS': 3,
#              'CHOOSING A COLLEGE': 4, 'AIR POLLUTION': 5, 'GARDENING': 6, 'BOATING AND SAILING': 7,
#              'BASKETBALL': 8, 'CREDIT CARD USE': 9, 'LATIN AMERICA': 10, 'FAMILY LIFE': 11, 'METRIC SYSTEM': 12,
#              'BASEBALL': 13, 'TAXES': 14, 'BOOKS AND LITERATURE': 15, 'CRIME': 16, 'PUBLIC EDUCATION': 17,
#              'RIGHT TO PRIVACY': 18, 'AUTO REPAIRS': 19, 'MIDDLE EAST': 20, 'FOOTBALL': 21,
#              'UNIVERSAL PBLIC SERV': 22, 'CAMPING': 23, 'FAMILY FINANCE': 24, 'POLITICS': 25, 'SOCIAL CHANGE': 26,
#              'DRUG TESTING': 27, 'COMPUTERS': 28, 'BUYING A CAR': 29, 'WOODWORKING': 30, 'EXERCISE AND FITNESS': 31,
#              'GOLF': 32, 'CAPITAL PUNISHMENT': 33, 'NEWS MEDIA': 34, 'HOME REPAIRS': 35, 'PAINTING': 36,
#              'FISHING': 37, 'SOVIET UNION': 38, 'CHILD CARE': 39, 'IMMIGRATION': 40, 'JOB BENEFITS': 41,
#              'RECYCLING': 42, 'MUSIC': 43, 'TV PROGRAMS': 44, 'ELECTIONS AND VOTING': 45, 'FEDERAL BUDGET': 46,
#              'MOVIES': 47, 'AIDS': 48, 'HOUSES': 49, 'VACATION SPOTS': 50, 'VIETNAM WAR': 51, 'CONSUMER GOODS': 52,
#              'RECIPES/FOOD/COOKING': 53, 'GUN CONTROL': 54, 'CLOTHING AND DRESS': 55, 'MAGAZINES': 56,
#              'SVGS & LOAN BAILOUT': 57, 'SPACE FLIGHT AND EXPLORATION': 58, "WOMEN'S ROLES": 59,
#              'PUERTO RICAN STTEHD': 60, 'TRIAL BY JURY': 61, 'ETHICS IN GOVERNMENT': 62, 'FAMILY REUNIONS': 63,
#              'RESTAURANTS': 64, 'UNIVERSAL HEALTH INS': 65}


# ## Dialogue act label encoding, MRDA
# {'S':0, 'B':1, 'D':2, 'F':3, 'Q':4}

# ## Dialogue act label encoding, DyDA
# {1:0, 2:1, 3:2, 4:3}

# ## Topic label encoding, DyDA
# {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}


tokenizer = AutoTokenizer.from_pretrained('roberta-base')
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')




class DataProcessor:

    @classmethod
    def read_data(cls, corpus, phase, context_len, chunk_size):
        df = pd.read_csv(f'../data/{corpus}/{phase}.csv', encoding='latin-1')
        max_conv_len = df['conv_id'].value_counts().max()
        # if phase != 'train':
        #     chunk_size = max_conv_len
        text_all = df['text'].tolist()
        input_ids_all = []
        attention_mask_all = []
        for text in text_all:
            text = text.strip().lower()
            input_ids = tokenizer.encode(text)
            attention_mask = [1] * len(input_ids)
            input_ids_all.append(input_ids)
            attention_mask_all.append(attention_mask)

        input_ids_all = np.array(input_ids_all)
        attention_mask_all = np.array(attention_mask_all)

        input_ids_ = []
        attention_mask_ = []
        labels_ = []
        chunk_lens_ = [] # the total input id length of a dialog chunk
        speaker_ids_ = []
        topic_labels_ = []
        rm_labels_=[]

        conv_ids = df['conv_id'].unique()
        for conv_id in tqdm(conv_ids, ncols=100, desc=f'processing {phase} data'):
            if conv_id == 7851 and corpus == 'dyda':
                continue
            mask_conv = df['conv_id'] == conv_id
            df_conv = df[mask_conv]
            input_ids = input_ids_all[mask_conv]
            attention_mask = attention_mask_all[mask_conv]
            speaker_ids = df_conv['speaker'].values
            labels = df_conv['act'].values
            topic_labels = df_conv['topic'].values

            if corpus == 'dyda':
                input_ids_.append(input_ids.tolist())
                attention_mask_.append(attention_mask.tolist())
                speaker_ids_.append(speaker_ids.tolist())
                labels_.append(labels.tolist())
                topic_labels_.append(topic_labels.tolist())
                chunk_lens_.append(len(input_ids.tolist()))

                rm_label = np.zeros((len(speaker_ids), len(speaker_ids)), dtype=np.long)
                for i in range(len(speaker_ids)):
                    for j in range(len(speaker_ids)):
                        if abs(i - j) <= context_len:
                            rm_label[i, j] = 1
                        else:
                            rm_label[i, j] = 0
                rm_labels_.append(rm_label)

            else:
                chunk_indices = list(range(0, df_conv.shape[0], chunk_size)) + [df_conv.shape[0]]  # 把一轮对话按chunk_size划分
                for i in range(len(chunk_indices) - 1):
                    idx1, idx2 = chunk_indices[i], chunk_indices[i + 1]

                    chunk_input_ids = input_ids[idx1: idx2].tolist()
                    chunk_attention_mask = attention_mask[idx1: idx2].tolist()
                    chunk_labels = labels[idx1: idx2].tolist()
                    chunk_speaker_ids = speaker_ids[idx1: idx2].tolist()
                    chunk_topic_labels = topic_labels[idx1: idx2].tolist()
                    chunk_len = idx2 - idx1

                    input_ids_.append(chunk_input_ids)
                    attention_mask_.append(chunk_attention_mask)
                    labels_.append(chunk_labels)
                    chunk_lens_.append(chunk_len)
                    speaker_ids_.append(chunk_speaker_ids)
                    topic_labels_.append(chunk_topic_labels)

                    rm_label = np.zeros((chunk_len, chunk_len), dtype=np.long)
                    for i in range(chunk_len):
                        for j in range(chunk_len):
                            if abs(i - j) <= context_len:
                                rm_label[i, j] = 1
                            else:
                                rm_label[i, j] = 0
                    rm_labels_.append(rm_label)


            # print("here:", conv_id)

        return input_ids_, attention_mask_, labels_, chunk_lens_, speaker_ids_, topic_labels_, rm_labels_



class DialogueActData(Dataset):
    def __init__(self, corpus, phase, context_len, chunk_size=0):
        input_ids_, attention_mask_, labels_, chunk_lens_, speaker_ids_, topic_labels_, rm_labels_ = DataProcessor.read_data(corpus, phase, context_len, chunk_size)

        self.input_ids = input_ids_
        self.attention_mask = attention_mask_
        self.labels = labels_
        self.chunk_lens = chunk_lens_
        # self.speaker_ids = speaker_ids_
        # self.topic_labels = topic_labels_
        # self.rm_labels_ = rm_labels_

    def __getitem__(self, index):
        item = {
            'input_ids': self.input_ids[index],  # [chunk_size, max_len]
            'attention_mask': self.attention_mask[index],  # [chunk_size, max_len]
            'labels': self.labels[index],  # [chunk_size]
            'chunk_lens': self.chunk_lens[index],  # [1]
            # 'speaker_ids': self.speaker_ids[index],  # [chunk_size]
            # 'topic_labels': self.topic_labels[index],  # [chunk_size]
            # 'rm_labels':self.rm_labels_[index]
        }
        return item

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):  # batch是字典的列表
    output = {}
    max_length = max([len(tmp) for x in batch for tmp in x['input_ids']])
    chunk_size = max([len(x['input_ids']) for x in batch])
    input_ids = []
    attention_mask = []
    labels = []
    topic_labels = []
    speaker_ids = []
    utterance_attention_mask = []
    rm_labels = []
    for x in batch:
        input_ids_pad = [[tokenizer.pad_token_id] * max_length for _ in range(chunk_size - len(x['input_ids']))]
        attention_mask_pad = [[0] * max_length for _ in range(chunk_size - len(x['attention_mask']))]
        input_ids_tmp = [tmp + [tokenizer.pad_token_id] * (max_length - len(tmp)) for tmp in x['input_ids']] + input_ids_pad
        attention_mask_tmp = [tmp + [0] * (max_length - len(tmp)) for tmp in x['attention_mask']] + attention_mask_pad
        labels_tmp = x['labels'] + [-1] * (chunk_size - len(x['labels']))
        topic_labels_tmp = x['topic_labels'] + [-1] * (chunk_size - len(x['topic_labels']))
        speaker_ids_tmp = x['speaker_ids'] + [2] * (chunk_size - len(x['speaker_ids']))
        utterance_attention_mask_tmp = [1] * x['chunk_lens'] + [0] * (chunk_size - x['chunk_lens'])
        rm_labels_tmp = -1 * np.ones((chunk_size, chunk_size))
        rm_labels_tmp[:len(x['input_ids']), :len(x['input_ids'])] = x['rm_labels']

        input_ids.append(input_ids_tmp)
        attention_mask.append(attention_mask_tmp)
        labels.append(labels_tmp)
        topic_labels.append(topic_labels_tmp)
        speaker_ids.append(speaker_ids_tmp)
        utterance_attention_mask.append(utterance_attention_mask_tmp)
        rm_labels.append(rm_labels_tmp)

    output['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
    output['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
    output['labels'] = torch.tensor(labels, dtype=torch.long)
    output['topic_labels'] = torch.tensor(topic_labels, dtype=torch.long)
    output['speaker_ids'] = torch.tensor(speaker_ids, dtype=torch.long)
    output['chunk_lens'] = torch.tensor([x['chunk_lens'] for x in batch], dtype=torch.long)  # [B]
    output['utterance_attention_mask'] = torch.tensor(utterance_attention_mask, dtype=torch.long)
    output['topic_labels'] = torch.tensor(topic_labels, dtype=torch.long)  # [B]
    _tmp_rm_labels = torch.tensor(rm_labels, dtype=torch.long)  # [B, chunk_size, chunk_size]
    output['rm_labels'] = _tmp_rm_labels.reshape(-1)  # [B, chunk_size*chunk_size]

    return output

def data_loader(corpus, phase, batch_size, context_len, chunk_size=0, shuffle=False):
    dataset = DialogueActData(corpus, phase, context_len, chunk_size=chunk_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


if __name__ == '__main__':
    corpus = 'dyda'
    phase = 'test'
    chunk_size = 350
    context_len = 20
    input_ids_, attention_mask_, labels_, chunk_lens_, speaker_ids_, topic_labels_, rm_labels_ = DataProcessor.read_data(corpus, phase, context_len, chunk_size)
    topic_labels_ = [label for labels in topic_labels_ for label in labels]
    topic_labels = set(topic_labels_)
    print(len(topic_labels))
    print(topic_labels)
