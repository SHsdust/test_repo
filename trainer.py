import torch
import torch.nn as nn
import os
import numpy as np
from dataset import data_loader
from model import BertRNN
from sklearn.metrics import accuracy_score, f1_score
import copy
import os
import yaml
import logging
import random
from logging import StreamHandler, FileHandler
from transformers import get_linear_schedule_with_warmup
import sys
from transformers import AdamW
from tqdm import tqdm
from cm_plot_swda_qiji2 import plot_confusion_matrix2_swda
from cm_plot_mrda import plot_confusion_matrix2_mrda



logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_logger(log_filename):
    handler1 = StreamHandler(stream=sys.stdout)
    handler2 = FileHandler(filename=log_filename, mode='a', delay=False)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[handler1, handler2]
    )


class Trainer:
    def __init__(self, args):

        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        print(device)
        set_seed(args.seed)
        init_logger(args.log_filename)
        logger.info('=' * 20 + 'Job start !' + '=' * 20)
        logger.info(args)

        train_loader = data_loader(corpus=args.corpus,
                                   phase='train',
                                   batch_size=args.batch_size,
                                   context_len=args.context_len,
                                   chunk_size=args.chunk_size,
                                   shuffle=True) if args.mode != 'inference' else None
        val_loader = data_loader(corpus=args.corpus,
                                 phase='val',
                                 batch_size=args.batch_size_val,
                                 context_len=args.context_len,
                                 chunk_size=args.chunk_size) if args.mode != 'inference' else None
        test_loader = data_loader(corpus=args.corpus,
                                  phase='test',
                                  batch_size=args.batch_size_val,
                                  context_len=args.context_len,
                                  chunk_size=args.chunk_size)


        if torch.cuda.device_count() > 0:
            logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")

        logger.info('Initializing model....')
        logger.info(f'{args.nfinetune}')
        model = BertRNN(nlayer=args.nlayer,
                        nclass=args.nclass,
                        emb_batch=args.emb_batch,
                        dropout=args.dropout,
                        nfinetune=args.nfinetune,
                        # speaker_info=args.speaker_info,
                        # topic_info=args.topic_info,
                        # emb_batch=args.emb_batch,
                        )

        # NosiyTune
        # noise_lambda = 0.2
        # for name, para in model.named_parameters():
        #     model.state_dict()[name][:] += (torch.rand(para.size()) - 0.5) * noise_lambda * torch.std(para)

        # # model = nn.DataParallel(model)
        # if torch.cuda.device_count() >= 2:
        #     model = nn.DataParallel(model)

        model.to(device)
        params = model.parameters()

        optimizer = AdamW(params, lr=args.lr, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.args = args
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader



    def train(self):
        best_epoch = 0
        best_epoch_dev_acc = 0
        best_epoch_test_acc = 0
        best_acc = 0
        best_f1 = 0
        best_state_dict = copy.deepcopy(self.model.state_dict())
        for epoch in range(self.args.epochs):
            logger.info(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}")


            train_loss, train_acc, train_f1, train_f1a = self.train_epoch(epoch, self.args.epochs)
            dev_loss, dev_acc, dev_f1, dev_f1a = self.eval('valid')
            test_loss, test_acc, test_f1, test_f1a = self.eval('test')

            logger.info(
                "Epoch: {} Train Loss: {:.4f} F1: {:.4f} Acc: {:.4f} F1a: {:.4f}".format(epoch + 1, train_loss, train_f1,train_acc, train_f1a))
            logger.info(
                "Epoch: {} Dev Loss: {:.4f} F1: {:.4f} Acc: {:.4f} F1a: {:.4f}".format(epoch + 1, dev_loss, dev_f1, dev_acc, dev_f1a))
            logger.info(
                "Epoch: {} Test Loss: {:.4f} F1: {:.4f} Acc: {:.4f} F1a: {:.4f}".format(epoch + 1, test_loss, test_f1, test_acc, test_f1a))

            if dev_acc > best_epoch_dev_acc:
                best_epoch = epoch
                best_epoch_dev_acc = dev_acc
                best_epoch_test_acc = test_acc
                best_state_dict = copy.deepcopy(self.model.state_dict())
            if test_acc > best_acc:
                best_acc = test_acc

            logger.info(f'Best Epoch: {best_epoch + 1}, Best Epoch Val Acc: {best_epoch_dev_acc:.4f}, '
                f'Best Epoch Test Acc: {best_epoch_test_acc:.4f}, Best Test Acc: {best_acc:.4f}\n')
            if epoch - best_epoch >= 10000:
                break
        logger.info('Saving the best checkpoint....')
        if not os.path.exists('./ckp'):
            os.makedirs('./ckp')
        torch.save(best_state_dict, f"./ckp/model_{self.args.corpus}_{best_acc:.4f}.pt")
        self.model.load_state_dict(best_state_dict)
        loss, acc, f1, f1a = self.eval('test')
        logger.info(f'Test Acc: {acc:.4f}, f1{f1:.4f}, f1a{f1a:.4f}')




    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        epoch_ce_loss = 0.
        y_pred = []
        y_true = []
        for _, batch in enumerate(tqdm(self.train_loader, desc=f'Train epoch {epoch + 1} / {total_epochs}', ncols=100)):
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            chunk_lens = batch['chunk_lens']
            speakers_ids = batch['speaker_ids'].to(self.device)
            utterance_attention_mask = batch['utterance_attention_mask'].to(self.device)
            rm_labels = batch['rm_labels'].to(self.device)  # [B*chunk_size*chunk_size]

            outputs, ce_loss = self.model(input_ids, attention_mask, speakers_ids, chunk_lens, utterance_attention_mask, rm_labels, args.w_da, args.w_rr, args.w_rm, da_labels=labels)
            loss = ce_loss
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()
            epoch_ce_loss += ce_loss.item()
            y_pred.append(outputs.detach().to('cpu').argmax(dim=1).numpy())
            labels = labels.reshape(-1)
            y_true.append(labels.detach().to('cpu').numpy())

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        mask = y_true != -1
        acc = accuracy_score(y_true[mask], y_pred[mask])
        f1 = f1_score(y_true[mask], y_pred[mask], average='weighted')
        f1a = f1_score(y_true[mask], y_pred[mask], average='macro')
        return epoch_ce_loss / len(self.train_loader), acc, f1, f1a

    def eval(self, mode='valid', plot=False):
        self.model.eval()
        epoch_ce_loss = 0.
        y_pred = []
        y_true = []
        loader = self.test_loader
        with torch.no_grad():
            for _, batch in enumerate(tqdm(loader, desc=mode, ncols=100)):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                chunk_lens = batch['chunk_lens']
                speakers_ids = batch['speaker_ids'].to(self.device)
                utterance_attention_mask = batch['utterance_attention_mask'].to(self.device)
                rm_labels = batch['rm_labels'].to(self.device)  # [B*chunk_size*chunk_size]

                outputs, ce_loss = self.model(input_ids, attention_mask, speakers_ids, chunk_lens, utterance_attention_mask, rm_labels, args.w_da, args.w_rr, args.w_rm, da_labels=labels)
                epoch_ce_loss += ce_loss.item()
                y_pred.append(outputs.detach().to('cpu').argmax(dim=1).numpy())
                labels = labels.reshape(-1)
                y_true.append(labels.detach().to('cpu').numpy())

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        mask = y_true != -1
        acc = accuracy_score(y_true[mask], y_pred[mask])
        f1 = f1_score(y_true[mask], y_pred[mask], average='weighted')
        f1a = f1_score(y_true[mask], y_pred[mask], average='macro')
        if plot:
            if args.corpus=='swda':
                plot_confusion_matrix2_swda(y_true[mask], y_pred[mask], f"./ckp/model_swda_{acc:.4f}.pt")
                np.savetxt(f"./ckp/da_gold_swda_{acc:.4f}.txt", y_true[mask].T, fmt='%i', delimiter='\t')
                np.savetxt(f"./ckp/da_pred_swda_{acc:.4f}.txt", y_pred[mask].T, fmt='%i', delimiter='\t')
            else:
                plot_confusion_matrix2_mrda(y_true[mask], y_pred[mask], f"./ckp/model_mrda_{acc:.4f}.pt")
                np.savetxt(f"./ckp/da_gold_mrda_{acc:.4f}.txt", y_true[mask].T, fmt='%i', delimiter='\t')
                np.savetxt(f"./ckp/da_pred_mrda_{acc:.4f}.txt", y_pred[mask].T, fmt='%i', delimiter='\t')

        return epoch_ce_loss / len(self.train_loader), acc, f1, f1a

    def save_model(self, name):
        if not os.path.exists('./ckp'):
            os.makedirs('./ckp')
        torch.save(self.model.state_dict(), './ckp/model_'+name+'.pth')

    def load_model(self, path):
        # self.model.load_state_dict(torch.load(path)) ##on GPU
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))) ##on CPU

    def inference(self, acc):
        ## using the trained model to inference on a new unseen dataset

        # load the saved checkpoint
        # change the model name to whatever the checkpoint is named
        logger.info('=' * 20 + 'Start Test' + '=' * 20)
        path = f"./ckp/model_{args.corpus}_{acc:.4f}.pt"
        self.load_model(path)
        test_loss, test_acc, test_f1, test_f1a = self.eval('test', plot=True)
        logger.info(f'Test Acc: {test_acc:.4f}, Test f1: {test_f1:.4f}, Test f1a: {test_f1a:.4f}\n')


if __name__ == '__main__':
    import argparse

    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./config/swda.yaml', help='config file path')
    # parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--parallel_training', type=bool, default=True)
    parser.add_argument("--local_rank", type=int, default=-1)  # for providing gpu id for each progress
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--acc', type=float, default=0)
    args = parser.parse_args()

    with open(args.config_file, 'rb') as f: ##20230119windows跑实验 从‘r’修改为'rb'
        dict_load = yaml.load(f, Loader=yaml.FullLoader)

    args.corpus = dict_load['corpus']
    args.nclass = dict_load['nclass']
    args.batch_size = dict_load['batch_size']
    args.batch_size_val = dict_load['batch_size_val']
    args.emb_batch = dict_load['emb_batch']
    args.epochs = dict_load['epochs']
    args.lr = float(dict_load['lr'])
    args.nlayer = dict_load['nlayer']
    args.chunk_size = dict_load['chunk_size']
    args.dropout = dict_load['dropout']
    args.speaker_info = dict_load['speaker_info']
    args.topic_info = dict_load['topic_info']
    args.nfinetune = dict_load['nfinetune']
    args.seed = dict_load['seed']
    args.warmup_rate = float(dict_load['warmup_rate'])
    args.emb_batch = dict_load['emb_batch']
    args.context_len = dict_load['context_len']
    args.w_da = dict_load['w_da']
    args.w_rr = dict_load['w_rr']
    args.w_rm = dict_load['w_rm']

    config_path = '/'.join(args.config_file.split('/')[: -1])
    args.log_filename = os.path.join(config_path, args.corpus + '.log')

    logger.info(f'{args}')
    trainer = Trainer(args)
    if args.mode == 'train':
        trainer.train()
    else:
        trainer.inference(args.acc)
