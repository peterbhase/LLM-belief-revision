import torch
from torch.nn import CrossEntropyLoss
import sys, os
import transformers
import numpy as np

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import utils.utils
import utils.metrics

def generate_batch(model, tokenizer, input_ids, attention_mask, max_new_tokens, debug=False):
    batch_size = input_ids.shape[0]
    device = input_ids.device
    for step in range(max_new_tokens):
        # Get the logits for the next token
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        if debug:
            if step == 0:
                print("generating:")
                print([tokenizer.decode([idx]) for idx in input_ids.squeeze()])
            probs = torch.softmax(logits, dim=-1).squeeze()
            argidx = torch.argsort(probs)
            logits_and_probs = [(tokenizer.decode([i]), prob.item()) for i, prob in zip(argidx, logits.squeeze()[argidx])]
            toks_and_probs = [(tokenizer.decode([i]), prob.item()) for i, prob in zip(argidx, probs[argidx])]
            print("step: ", step)
            print(argidx[-5:])
            print(logits_and_probs[-5:])
            print(toks_and_probs[-5:])
        # Select the most likely token for each sequence in the batch
        next_tokens = torch.argmax(logits, dim=-1)
        # Append the most likely token to the input IDs
        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
        # Extend the attention mask
        attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)], dim=-1)
    return input_ids

def strip_right_padded_single_input(main_kwargs, pad_token_id):
    input_ids = main_kwargs["input_ids"]
    attention_mask = main_kwargs["attention_mask"]
    last_non_pad_index = (input_ids.squeeze() != pad_token_id).nonzero()[-1].item()
    main_kwargs["input_ids"] = input_ids[:, :last_non_pad_index + 1]
    main_kwargs["attention_mask"] = attention_mask[:, :last_non_pad_index + 1]
    if 'targets_mask' in main_kwargs:
        main_kwargs["targets_mask"] = main_kwargs["targets_mask"][:, :last_non_pad_index + 1]
    return main_kwargs

def decode_batched_sequences(tokenizer, input_ids, bos_token_id, eos_token_id):
    # Decode the generated sequences
    batch_size = input_ids.size(0)
    generated_sequences = []
    for i in range(batch_size):
        sequence = input_ids[i].tolist()
        if bos_token_id == eos_token_id:
            # Stop at the second occurrence of the EOS token
            eos_positions = [pos for pos, token_id in enumerate(sequence) if token_id == eos_token_id]
            if len(eos_positions) > 1:
                sequence = sequence[:eos_positions[1]]
        else:
            # Stop at the first occurrence of the EOS token
            if eos_token_id in sequence:
                sequence = sequence[:sequence.index(eos_token_id)]
        decoded_sequence = tokenizer.decode(sequence, skip_special_tokens=True)
        generated_sequences.append(decoded_sequence)
    return generated_sequences

def str_clean(data):
    if data is not None:
        return data.strip().lower()
    else:
        return None

def em_accuracy_sum(preds, labels, return_vec=False):
    assert len(preds) == len(labels)
    # strict calculation of accuracy for predictions from fewshot model
    preds = np.array([str_clean(x) for x in preds])
    labels = np.array([str_clean(label) for label in labels])
    correct = (preds==labels)
    if return_vec:
        return correct.sum(), correct
    else:
        return correct.sum()

def fewshot_accuracy_sum(preds, labels, possible_labels=None, return_vec=False):
    # generous calculation of accuracy for predictions from fewshot model
    # an answer is 'predicted' if it appears in the pred str
    # tie breaking is done randomly if the pred str mentions >1 label
    # returns acc sum, optionally the vector of binary 0/1 accs per point
    assert len(preds) == len(labels)
    n_correct = 0
    correct_indicators = []
    # clean arrays
    preds = np.array([str_clean(x) for x in preds])
    labels = np.array([str_clean(label) for label in labels])
    if possible_labels is not None:
        possible_labels = np.array([str_clean(x) for x in possible_labels])
    else:
        possible_labels = []
    # loop through preds and labels
    for pred, label in zip(preds, labels):
        # make label-specific possible_labels as needed
        if label not in possible_labels:
            possible_labels = [label, 'NO_ANSWER_DETECTED']
            answer_to_counts = {answer : 0 for answer in possible_labels}
        # first see if pred is exactly in answers
        if pred in possible_labels:
            answer_to_counts[pred] += 1
        # if not, then count how often labels appear inside of pred
        else:
            for answer in possible_labels:
                if answer in pred:
                    answer_to_counts[answer] += 1
        max_count = max(answer_to_counts.values())
        max_preds = [pred for pred in answer_to_counts.keys() if answer_to_counts[pred] == max_count]
        if len(max_preds) == 1:
            use_pred = max_preds[0]
        else:
            use_pred = 'NO_ANSWER_DETECTED'
        correct = (use_pred == label)
        n_correct += correct
        correct_indicators.append(correct)
    if not return_vec:
        return n_correct
    else:
        return n_correct, np.array(correct_indicators)

def compute_probs_from_batch(model, batch, return_value='log_probs', pad_token_id=None, tokenizer=None):
    '''
    Compute target probabilities for batch
    - batch can be obtained with make_LM_batch(**kwargs) or tokenizer(strs)
    '''
    assert return_value in ['probs', 'log_probs', 'log_probs_seq_avg'] 
    model_batch = {
        'input_ids' : batch['input_ids'],
        'attention_mask' : batch['attention_mask']
    }
    target_tokens = batch['input_ids']
    if 'targets_mask' in batch and batch['targets_mask'] is not None:
        target_mask = batch['targets_mask']
    else:
        assert pad_token_id is not None
        target_mask = target_tokens != pad_token_id
    outputs = model(**model_batch)
    logits = outputs.logits
    loss_fct = CrossEntropyLoss(reduction='none')
    shift_logits = logits[..., :-1, :]
    shift_labels = target_tokens[..., 1:]
    shift_mask = target_mask[...,1:]
    nll = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
    nll = nll.reshape(logits.size(0), -1) # batch size x num tokens
    if return_value == 'log_probs':
        nll = (shift_mask * nll).sum(-1) # sum over token dimension
        probs = -nll # one value per sequence
    elif return_value == 'probs':
        nll = (shift_mask * nll).sum(-1) # sum over token dimension
        probs = torch.exp(-nll) # one probability per input sequence
    # reshape probs to be num_items x num_answers
    probs = probs.reshape(len(probs), 1)
    return probs

def make_LM_batch(tokenizer, prompts, label_strs, padding_side='left', add_eos_token=False,
                  max_len=None, generative_batch=False, extend_padding_by=False):
    '''
    This makes inputs for computing LM probabilities of labels given prompts, when generative_batch=False
    e.g. for prompts = ["I like", "I like", "I do not like", "I do not like"] and answer_choices = ["dogs", "cats, "birds", "fish]
        with labels = [1,1] (repeating of prompts and flattening of nested answer choice list expected to be done prior to this method)
    This returns a dict
    {
        "input_ids": tokenized ["I like dogs", "I like cats", "I do not like birds", "I do not like fish"]
        "attention_mask": normal attention mask for the tokenizer
        "targets_mask": tensor, 0 where a token belonged in the prompt, 1 where it belonged in answer_choice, 0 for padding
    }
    intended for use with compute_probs_from_batch
    args:
        prompts: model input for generation, or part of sequence for which loss is NOT calculated when generative_batch=False
        labels: label for generation, or part of sequence for which loss IS calculated when generative_batch=False
        generative_batch: when true, input_ids does not contain both prompts and answer_choices
        extend_padding_by: used to extend padding while max_len=None (see use in dataloaders.py)
    '''
    # set pad and bos tokens as needed
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id # note we never supervise model with eos token
    if tokenizer.bos_token_id is None:
        bos_token_list = []
    else:
        bos_token_list = [tokenizer.bos_token_id]
    # tokenize inputs. whether or not there should be a space before the answer depends on the model tokenizer
    no_space_before_answer = any([x in str(type(tokenizer)).lower()] for x in ['mistral'])
    if prompts is not None:
        prompt_ids = [tokenizer.encode(f"{prompt}", add_special_tokens=False) for prompt in prompts]
        if no_space_before_answer:
            label_ids = [tokenizer.encode(f"{answer}", add_special_tokens=False) for answer in label_strs]
        else:
            label_ids = [tokenizer.encode(f" {answer}", add_special_tokens=False) for answer in label_strs]
    else:
        prompt_ids = [[] for _ in range(len(label_strs))]
        label_ids = [tokenizer.encode(f"{seq}", add_special_tokens=False) for seq in label_strs]
    if generative_batch:
        lm_inputs = [bos_token_list + _prompt_ids for _prompt_ids in prompt_ids]
    else:
        lm_inputs = [bos_token_list + _prompt_ids + _label_ids for _prompt_ids, _label_ids in zip(prompt_ids, label_ids)]
    # add eos tokens if requested
    if add_eos_token:
        lm_inputs = [x + [tokenizer.eos_token_id] for x in lm_inputs]
    # pad lm inputs
    if max_len is not None and max_len > 0:
        # assert not max([len(input_ids) for input_ids in lm_inputs]) > max_len, f"Trying to make LM batch with inputs that are too long for max len {max_len}"
        for no, _input_ids in enumerate(lm_inputs):
            if len(_input_ids) > max_len:
                sent_ids = prompt_ids[no] + label_ids[no]
                shorten_by = len(_input_ids) - max_len
                # sent = tokenizer.decode(sent_ids)
                # print(f" -- Shortening input \'{sent}\' from {len(_input_ids)} toks to {max_len} toks")
                lm_inputs[no] = _input_ids[:max_len]
                # shorten corresponding prompt tokens
                num_tokens = len(prompt_ids[no]) + len(label_ids[no]) + add_eos_token + (tokenizer.bos_token_id is not None)
                while num_tokens > max_len:
                    _prompt = prompt_ids[no]
                    _label = label_ids[no]
                    shorten_prompt = len(_prompt) > 0 # prefer to shorten prompt rather than labels
                    if shorten_prompt:
                        new_len = len(_prompt) - shorten_by
                        prompt_ids[no] = prompt_ids[no][:new_len]
                    else:
                        new_len = len(_label) - shorten_by
                        label_ids[no] = label_ids[no][:new_len]
                    num_tokens = len(prompt_ids[no]) + len(label_ids[no]) + add_eos_token + (tokenizer.bos_token_id is not None)
        use_max_len = max_len
    else:
        use_max_len = max([len(input_ids) for input_ids in lm_inputs])
    if extend_padding_by is not None:
        use_max_len += extend_padding_by
    # left-pad inputs to max len of batch
    for lm_input in lm_inputs:
        short_by = use_max_len - len(lm_input)
        if padding_side == 'left':
            lm_input[:0] = [tokenizer.pad_token_id]*short_by # somehow this is proper indexing...
        else:
            lm_input += [tokenizer.pad_token_id]*short_by 
    # now get label masks
    if generative_batch:
        targets_mask = None
    else:
        targets_mask = []
        for _prompt_ids, _label_ids in zip(prompt_ids, label_ids):
            num_tokens = len(_prompt_ids) + len(_label_ids) + add_eos_token + (tokenizer.bos_token_id is not None)
            num_target_tokens = len(_label_ids) + add_eos_token
            if padding_side == 'left':
                label_mask = [0]*(use_max_len-num_target_tokens) + [1]*(num_target_tokens)
            elif padding_side == 'right':
                label_mask = [0]*(num_tokens-num_target_tokens) + [1]*num_target_tokens + [0]*(use_max_len-num_tokens)
            targets_mask.append(label_mask)
        targets_mask = torch.tensor(targets_mask)
    # and an attention mask
    lm_inputs = torch.tensor(lm_inputs)
    attention_mask = lm_inputs != tokenizer.pad_token_id
    batch = {
        'input_ids': lm_inputs,
        'attention_mask': attention_mask,
        'targets_mask': targets_mask,
    }
    if not generative_batch:
        batch['labels'] = lm_inputs
    else:
        batch['labels'] = label_ids
    return batch

def make_generation_prompts(tokenizer, inputs):
    prompt_ids = [tokenizer.encode(" " + prompt, add_special_tokens=False) for prompt in inputs]
    prompt_ids = [[tokenizer.bos_token_id] + prompt_ids for prompt_ids in prompt_ids]
    # pad lm inputs
    max_len = max([len(prompt) for prompt in prompt_ids])
    for prompt in prompt_ids:
        short_by = max_len - len(prompt)
        # prepend padding (left-handed padding)
        prompt[:0] = [tokenizer.pad_token_id]*short_by
    prompt_ids = torch.tensor(prompt_ids)
    batch = {
        'input_ids': prompt_ids,
        'attention_mask': (prompt_ids != tokenizer.pad_token_id)
    }
    return batch

def remove_input_prefix_from_generations(tokenizer, preds, prompts):
    """
    model generations include the prompts by default. this removes these from the generation
    """
    if type(preds) is torch.Tensor:
        preds = decode_batched_sequences(tokenizer, preds, tokenizer.bos_token_id, tokenizer.eos_token_id)
        # preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in preds]
    if type(prompts) is torch.Tensor:
        prompts = decode_batched_sequences(tokenizer, prompts, tokenizer.bos_token_id, tokenizer.eos_token_id)
        # prompts = [tokenizer.decode(x, skip_special_tokens=True) for x in prompts]
    assert len(preds) == len(prompts)
    preds = [pred.replace(prompt, "") for pred, prompt in zip(preds, prompts)]
    return preds

