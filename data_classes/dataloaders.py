import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import LM_utils
import data_utils
import utils
from utils import min_max_mean
from data_utils import verbalize_fact, promptify_fact, state_P_is_true_false, connect_sentences, get_random_fact, get_false_fact
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from copy import deepcopy
import json


class TextDataset(Dataset):
    # simple text dataset for use in PropositionDataset
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PropositionDataset():
    '''
    This class is used for creating dataloaders for training and evaluation.
    The fundamental unit of the training dataloader is the document, which is a dict containing a str to train on with an autoregressive loss
    The fundamental unit of the eval dataloader is the test case, which is a dict with many input prompts and labels to evaluate the model on
    We define special collate_fn that is passed to a constructed dataloader, depending on if the split is train or eval
    - datasets are read/loaded in get_train_dataloader and get_eval_dataloader
    '''
    def __init__(
        self,
        args,
        tokenizer,
        train_or_eval,
    ):
        super().__init__()
        assert train_or_eval in ['train', 'eval']
        self.args = args
        self.tokenizer = tokenizer
        self.rng = np.random.default_rng(args.seed)
        # assert tokenizer.padding_side == 'left', "must use left padding for tokenization at all times"
        assert tokenizer.bos_token
        assert tokenizer.eos_token
        if not self.tokenizer.pad_token_id:
            self.tokenizer.add_special_tokens({'pad_token' : tokenizer.decode([0])})
        
    def retrieve_ids(self, idx):
        data = [self.all_docs[_id] for _id in idx]
        return data

    def get_random_subset(self, dataloader, size, exclude_ids, batch_size, data_sample_counts=None):
        n_points = len(dataloader.dataset)
        eligible_idx = np.setdiff1d(np.arange(n_points), np.array(exclude_ids).reshape(-1))
        sample_idx = np.random.choice(eligible_idx, min(size, len(eligible_idx)), replace=False)
        subset = torch.utils.data.Subset(dataloader.dataset, sample_idx.tolist())
        subset_dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, collate_fn=dataloader.collate_fn, shuffle=False)
        return subset_dataloader, sample_idx
    
    def compute_num_tokens_in_batch(self, batch):
        input_ids = batch['input_ids']
        non_special_tokens = (input_ids != self.tokenizer.pad_token_id) * (input_ids != self.tokenizer.eos_token_id) * (input_ids != self.tokenizer.bos_token_id)
        attended_non_special_tokens = non_special_tokens * batch['attention_mask']
        num_tokens = attended_non_special_tokens.sum()
        return num_tokens.item()

    def train_collate_fn(self, items):
        '''
        used in pytorch dataloader, this function tokenizes a batch
        args:
            items: list of sentences/documents for processing
        returns:
            dict with tensor inputs, labels and orig strings
        '''
        doc_strs = []
        num_sentences = []
        for doc_dict in items:
            doc_str = doc_dict['document']
            doc_strs.append(doc_str)
            num_sentences.append(doc_dict['num_sentences'])
        # tokenize inputs with max_seq_len. we force pad to max_seq_len if provided
        if self.args.random_traintime_padding:
            extend_padding_by = self.rng.choice(list(range(2*self.args.max_object_tokens)))
        else:
            extend_padding_by = None
        batch = LM_utils.make_LM_batch(self.tokenizer, prompts=None, label_strs=doc_strs, padding_side=self.args.padding_side, add_eos_token=True, max_len=self.args.max_seq_len, 
                                       generative_batch=False, extend_padding_by=extend_padding_by)
        batch['metadata'] = {
            'documents': doc_strs,
            'num_sentences': sum(num_sentences),
            'num_tokens': self.compute_num_tokens_in_batch(batch),
        }
        return batch

    def eval_collate_fn(self, items):
        '''
        Used in pytorch dataloader, this function tokenizes a batch
        '''
        # first preprocess to get prompts and labels
        per_item_metadata = []
        per_case_metadata = [] 
        prompts = []
        targets = []
        doc_dicts = []
        ids = []
        target_n_tokens = []
        str_label_dicts = []
        for doc_dict in items:
            # for each case, we want to get a promptified version for generative eval and an LM version for computing the probability of the target
            item_id = doc_dict['metadata']['id']
            str_labels = {}
            for case, test_case in doc_dict.items():
                if case == 'metadata':
                    ids.append(item_id)
                    per_item_metadata.append(test_case)
                    continue
                prompts.append(test_case['model_input'])
                targets.append(test_case['label'])
                metadata = {'case': case, 
                            'id': item_id,
                            'seen_during_training': test_case['seen_during_training'], 
                            'sentence_is_true': test_case['sentence_is_true'],
                            'use_for_acc_eval': test_case['use_for_acc_eval']}
                per_case_metadata.append(metadata)
                target_n_tokens.append(len(self.tokenizer.encode(test_case['label'], add_special_tokens=False)))
                prob_targets = {k:v for k,v in test_case.items() if 'update_prob' in k or 'prob' in k}
                metadata.update(prob_targets)
                str_labels[case] = test_case['label']
            doc_dicts.append(doc_dict)
            str_label_dicts.append(str_labels)
        # tokenize inputs for generative eval and LM prob eval
        if self.args.padding_side == 'left' and self.args.eval_batch_size == 1 and self.args.max_seq_len >=1:
            # take off the answer length to maintain the original alignment that would have been seen during training. I thought we should -1 too in both cases...but the below approach leads to exact alignment
            gen_max_len = self.args.max_seq_len - 1 - target_n_tokens[0]
            LM_max_len = self.args.max_seq_len - 1 
            # check if there's a difference in tokening prompt+target together vs prompt
            sentence = [doc_dict['metadata']['all_sentences'][0] for doc_dict in doc_dicts[:1]][0]
            qa_toks = self.tokenizer.encode(sentence)
            q_toks = self.tokenizer.encode(prompts[0])
            a_toks = self.tokenizer.encode(targets[0])
            if len(qa_toks) != len(q_toks) + len(a_toks):
                qa_longer_by = len(qa_toks) - (len(q_toks) + len(a_toks))
                gen_max_len -= qa_longer_by
            LM_max_len = self.args.max_seq_len - 1
            gen_batch = LM_utils.make_LM_batch(self.tokenizer, prompts, targets, padding_side='left', max_len=gen_max_len, generative_batch=True)
            LM_batch = LM_utils.make_LM_batch(self.tokenizer, prompts, targets, padding_side='left', max_len=LM_max_len, add_eos_token=False, generative_batch=False)
            doc_strs = [prompt + " " + label for prompt, label in zip(prompts, targets)]
        elif self.args.padding_side == 'left':
            gen_batch = LM_utils.make_LM_batch(self.tokenizer, prompts, targets, padding_side='left', max_len=self.args.max_seq_len, generative_batch=True)
            LM_batch = LM_utils.make_LM_batch(self.tokenizer, prompts, targets, padding_side='left', max_len=self.args.max_seq_len, generative_batch=False)
        elif self.args.padding_side == 'right':
            gen_batch = LM_utils.make_LM_batch(self.tokenizer, prompts, targets, padding_side=self.args.padding_side, generative_batch=True)
            LM_batch = LM_utils.make_LM_batch(self.tokenizer, prompts, targets, padding_side=self.args.padding_side, generative_batch=False)
        # make a sequence modeling batch containing random sentences to check what the full-sequence LM loss would be
        doc_strs = [self.rng.choice(doc_dict['metadata']['all_sentences']) for doc_dict in doc_dicts]
        sequence_modeling_batch = LM_utils.make_LM_batch(self.tokenizer, prompts=None, label_strs=doc_strs, padding_side=self.args.padding_side, add_eos_token=True, max_len=self.args.max_seq_len, generative_batch=False)
        # return batch
        return_batch = {
            'gen_batch': gen_batch, # len = # cases
            'LM_batch': LM_batch, # len = # cases
            'sequence_modeling_batch': sequence_modeling_batch, # len = # cases
            'per_case_metadata': per_case_metadata, # len = # cases
            'per_item_metadata': per_item_metadata, # len = # items
            'doc_dicts': doc_dicts, # len = # cases
            'ids': ids,  # len = # cases
            'str_labels': utils.combine_list_of_dicts(str_label_dicts)
        }
        return return_batch
    
    def get_train_dataloader(self, args, data_path, shuffle=True):
        all_train_docs = []
        with open(data_path, 'r') as file:
            for line in file:
                train_doc = json.loads(line)
                all_train_docs.append(train_doc)
        dataset = TextDataset(all_train_docs)
        train_collate_fn = self.train_collate_fn
        dataloader = DataLoader(dataset, shuffle=shuffle, collate_fn=train_collate_fn, pin_memory=False, num_workers=0, batch_size=args.train_batch_size)
        return dataloader

    def get_eval_dataloader(self, args, data_path, n=None):
        # differs from train_dataloader via collate_fn
        # optionally, subsample the dataset
        all_eval_docs = []
        with open(data_path, 'r') as file:
            for line in file:
                eval_case = json.loads(line)
                all_eval_docs.append(eval_case)
        if n is None:
            use_eval_docs = all_eval_docs
        else:
            random_idx = np.random.choice(np.arange(n), size=n, replace=False)
            use_eval_docs = [all_eval_docs[i] for i in random_idx]
        dataset = TextDataset(use_eval_docs)
        eval_collate_fn = self.eval_collate_fn
        dataloader = DataLoader(dataset, shuffle=False, collate_fn=eval_collate_fn, pin_memory=False, num_workers=0, batch_size=args.eval_batch_size)
        return dataloader