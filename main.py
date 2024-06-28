import argparse
from copy import deepcopy
import numpy as np
import pandas as pd
import time
import os
import sys

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils.utils import str2bool, get_experiment_name
from utils.training_logger import TrainingLogger

def evaluate_model_editing(args, model, batch, tokenizer):
    '''
    Evaluate model editing on new_fact_case points in batch
    - we record probabilistic and logical coherence for each test case before and after updating on a new fact, which either reinforces or contradicts the main s r o input in the test case
    '''
    editing_stats_df = pd.DataFrame()
    data_ids = batch['ids']
    for data_id in data_ids:
        point_batch = utils.subset_batch_by_data_ids(batch, [data_id])
        step = 0
        data_point_stats = {
            'step': step,
            'id': data_id,
            'e1': point_batch['per_item_metadata'][0]['base_fact']['e1'],
            'rel': point_batch['per_item_metadata'][0]['base_fact']['rel'],
            'e2': point_batch['per_item_metadata'][0]['base_fact']['e2'],
            'requested_obj': point_batch['doc_dicts'][0]['editing_same_s_same_r_fact_requested_obj']['label'],
            'known_entity': point_batch['per_item_metadata'][0]['known_entity'],
            'has_relevant_property': point_batch['per_item_metadata'][0]['has_relevant_property'],
            'diff_r_is_dependent_rel': point_batch['per_item_metadata'][0]['diff_r_is_dependent_rel'],
            'same_r_is_dependent_rel': point_batch['per_item_metadata'][0]['same_r_is_dependent_rel'],
            'all_cases_independent': point_batch['per_item_metadata'][0]['all_cases_independent'],
            'using_diff_s_with_both_r_s': point_batch['per_item_metadata'][0]['using_diff_s_with_both_r_s'],
            'new_fact_reinforces': point_batch['per_item_metadata'][0]['new_fact_reinforces'],
            'downstream_answer_changes': point_batch['per_item_metadata'][0]['downstream_answer_changes'],
            'same_s_same_r_rel': point_batch['doc_dicts'][0]['editing_same_s_same_r_fact_orig_GT_obj']['rel'],
            'same_s_same_r_obj': point_batch['doc_dicts'][0]['editing_same_s_same_r_fact_orig_GT_obj']['label'],
            'diff_s_same_r_subj': point_batch['doc_dicts'][0]['editing_diff_s_same_r_fact_orig_GT_obj']['subj'],
            'diff_s_same_r_rel': point_batch['doc_dicts'][0]['editing_diff_s_same_r_fact_orig_GT_obj']['rel'],
            'diff_s_same_r_obj': point_batch['doc_dicts'][0]['editing_diff_s_same_r_fact_orig_GT_obj']['label'],
            'same_s_diff_r_subj': point_batch['doc_dicts'][0]['editing_same_s_diff_r_fact_orig_GT_obj']['subj'],
            'same_s_diff_r_rel': point_batch['doc_dicts'][0]['editing_same_s_diff_r_fact_orig_GT_obj']['rel'],
            'same_s_diff_r_obj': point_batch['doc_dicts'][0]['editing_same_s_diff_r_fact_orig_GT_obj']['label'],
            'diff_s_diff_r_subj': point_batch['doc_dicts'][0]['editing_diff_s_diff_r_fact_orig_GT_obj']['subj'],
            'diff_s_diff_r_rel': point_batch['doc_dicts'][0]['editing_diff_s_diff_r_fact_orig_GT_obj']['rel'],
            'diff_s_diff_r_obj': point_batch['doc_dicts'][0]['editing_diff_s_diff_r_fact_orig_GT_obj']['label'],
            'same_s_same_r_obj': point_batch['doc_dicts'][0]['editing_same_s_same_r_fact_orig_GT_obj']['label'],
            'same_s_same_r_new_modal_obj': point_batch['doc_dicts'][0]['editing_same_s_same_r_fact_new_modal_obj']['label'],
            'diff_s_same_r_new_modal_obj': point_batch['doc_dicts'][0]['editing_diff_s_same_r_fact_new_modal_obj']['label'],
            'same_s_diff_r_new_modal_obj': point_batch['doc_dicts'][0]['editing_same_s_diff_r_fact_new_modal_obj']['label'],
            'diff_s_diff_r_new_modal_obj': point_batch['doc_dicts'][0]['editing_diff_s_diff_r_fact_new_modal_obj']['label'],
        }
        # add same/diff s/r with req obj for all update points
        # unpack gen and LM batches
        gen_batch = point_batch['gen_batch']
        LM_batch = point_batch['LM_batch']
        
        batch_preds = get_batch_preds(args, model, tokenizer, point_batch)
        batch_probs = get_batch_probs(args, model, point_batch)

        # score the generations for generative accuracy
        n_correct, binary_correct = metrics.compute_acc_sum(tokenizer, batch_preds, gen_batch['labels'], return_where_correct=True)

        # get key statistic dicts
        per_case_metadata = point_batch['per_case_metadata']
        correctness_dict, seen_dict, probabilities_dict, outputs_dict = get_stat_dicts(batch, binary_correct, batch_probs, batch_preds)
        # calculate probabilistic coherence metrics for the pre update target, across cases
        prob_stats = get_prob_metrics(per_case_metadata, probabilities_dict) # CURRENTLY NOT SAVING THESE. SAVING ALL TARGETS INSTEAD
        # get logical coherence for the eval cases that are included with the test case
        logic_stats = get_logic_metrics(args, batch, probabilities_dict)
        probabilities_dict = utils.flip_probs_to_prob_true(probabilities_dict, batch['str_labels']) # NEED TO FLIP PROBS to p(true) to store in data_point_stats

        # add correctness and model output probs
        data_point_stats.update(correctness_dict)
        data_point_stats.update(probabilities_dict)
        data_point_stats.update(logic_stats)
        data_point_stats.update(outputs_dict)
        # add prob targets to data_point_stats
        
        for case_metadata in per_case_metadata:
            for target_name in ['pre_update_probs', 'post_update_probs_n=1000', 'post_update_probs_p=95']:
                if target_name in case_metadata:
                    for prob_type, prob in case_metadata[target_name].items():
                        if type(prob) is float:
                            case_name = case_metadata['case']
                            col_name = f"{case_name}_{target_name}_{prob_type}"
                            data_point_stats[col_name] = prob
        # data_point_stats.update(prob_stats) # DONT add prob_stats. will compute after the fact in plotting

        # acculumate stats
        point_df = pd.DataFrame(data_point_stats)
        editing_stats_df = pd.concat([editing_stats_df, point_df], ignore_index=True)

        # edit model
        edit_model = utils.PEFT_wrap_model(args, model)
        # load optimizer
        optimizer, scheduler = utils.load_update_optimizer(args, model)
        scheduler.step() # step to set constant LR

        # get the individual update input/output
        case_names = np.array([metadata['case'] for metadata in per_case_metadata])
        where_new_fact_idx = np.argwhere(case_names == 'new_fact_case').reshape(-1)
        update_point = utils.slice_batch_kwargs(LM_batch, where_new_fact_idx)
        # print the requested update info
        point_metadata = point_batch['per_item_metadata'][0]
        new_fact_doc = point_batch['doc_dicts'][0]['new_fact_case']
        orig_fact = point_metadata['base_fact']
        if data_id < args.num_print:
            print()
        print(f"Model editing point {data_id} | Requested update: {orig_fact['e1']} {orig_fact['rel']} {orig_fact['e2']} --> {new_fact_doc['label']}  | update reinforces: {point_metadata['new_fact_reinforces']} --------------------------------------------", end = '\r')
        if data_id < args.num_print:
            print()
        main_kwargs = {
                'input_ids': update_point['input_ids'],
                'attention_mask': update_point['attention_mask'],
                'targets_mask': update_point['targets_mask'],
                'labels': update_point['labels'],
            }
        n_tokens = utils.compute_num_tokens_in_batch(main_kwargs, tokenizer)
        utils.move_kwargs_to_gpu(main_kwargs)
        log_probs = LM_utils.compute_probs_from_batch(model, main_kwargs, return_value = 'log_probs', pad_token_id=tokenizer.pad_token_id)
        loss = -log_probs.sum() / n_tokens # avg per token
        req_prob = batch_probs[where_new_fact_idx.item()]
        model_output = batch_preds[where_new_fact_idx.item()]
        req_correct = binary_correct[where_new_fact_idx.item()]
        if data_id < args.num_print:
            print(f" step {step} | target loss: {loss:.4f} | target prob: {req_prob:.4f} | model output: {model_output} | output requested? {req_correct}")

        for step in range(1, args.update_steps+1):
            # print( "Update", end='\r' if not last_batch else '\n')
            with torch.enable_grad():
                log_probs = LM_utils.compute_probs_from_batch(model, main_kwargs, return_value = 'log_probs', pad_token_id=tokenizer.pad_token_id)
                loss = -log_probs.sum() / n_tokens # avg per token
                loss.backward()
                torch.nn.utils.clip_grad_norm_(edit_model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
                # evaluate model at this step
                batch_preds = get_batch_preds(args, model, tokenizer, point_batch)
                batch_probs = get_batch_probs(args, model, point_batch)
                n_correct, binary_correct = metrics.compute_acc_sum(tokenizer, batch_preds, gen_batch['labels'], return_where_correct=True)
                correctness_dict, seen_dict, probabilities_dict, outputs_dict = get_stat_dicts(batch, binary_correct, batch_probs, batch_preds)
                logic_stats = get_logic_metrics(args, batch, probabilities_dict)
                probabilities_dict = utils.flip_probs_to_prob_true(probabilities_dict, batch['str_labels']) # NEED TO FLIP PROBS to p(true) to store in data_point_stats

                # add correctness and model output probs
                data_point_stats['step'] = step
                data_point_stats.update(correctness_dict)
                data_point_stats.update(probabilities_dict)
                data_point_stats.update(logic_stats)
                data_point_stats.update(outputs_dict)
                req_prob = batch_probs[where_new_fact_idx.item()]
                model_output = batch_preds[where_new_fact_idx.item()]
                req_correct = binary_correct[where_new_fact_idx.item()]
                # print progress
                if data_id < args.num_print and step % 10 == 0:
                    print(f" step {step} | target loss: {loss:.4f} | target prob: {req_prob:.4f} | model output: {model_output} | output requested? {req_correct}")
                # accumulate result at step
                data_point_stats['last_step'] = (step == args.update_steps)
                point_df = pd.DataFrame(data_point_stats)
                editing_stats_df = pd.concat([editing_stats_df, point_df], ignore_index=True)

        # reset peft model
        model = edit_model.unload()

    return editing_stats_df

        
def get_batch_preds(args, model, tokenizer, batch):
    batch_preds = []
    gen_batch = batch['gen_batch']
    all_batch_idx = np.arange(len(batch['gen_batch']['input_ids']))
    for batch_idx in utils.chunk_array(all_batch_idx, size=args.eval_batch_size):
        main_kwargs = {
            'input_ids': gen_batch['input_ids'][batch_idx],
            'attention_mask': gen_batch['attention_mask'][batch_idx],
            'do_sample': False,
            'max_new_tokens': args.max_object_tokens,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id
        }
        utils.move_kwargs_to_gpu(main_kwargs)
        with torch.no_grad():
            # strip RHS padding if needed
            if args.padding_side == 'right': 
                assert args.eval_batch_size == 1, "Can only generate for one input at a time when using right padding"
                main_kwargs = LM_utils.strip_right_padded_single_input(main_kwargs, tokenizer.pad_token_id)
            preds = LM_utils.generate_batch(model, tokenizer, main_kwargs['input_ids'], main_kwargs['attention_mask'], max_new_tokens=args.max_object_tokens)
            preds = LM_utils.remove_input_prefix_from_generations(tokenizer, preds, main_kwargs['input_ids'])
            batch_preds.extend(preds)
    return batch_preds

def get_batch_probs(args, model, batch):
    batch_probs = []
    LM_batch = batch['LM_batch']
    all_batch_idx = np.arange(len(batch['LM_batch']['input_ids']))
    for batch_idx in utils.chunk_array(all_batch_idx, size=args.eval_batch_size):
        main_kwargs = {
            'input_ids': LM_batch['input_ids'][batch_idx],
            'attention_mask': LM_batch['attention_mask'][batch_idx],
            'targets_mask': LM_batch['targets_mask'][batch_idx],
            'labels': LM_batch['labels'][batch_idx]
        }
        utils.move_kwargs_to_gpu(main_kwargs)
        with torch.no_grad():
            probs = LM_utils.compute_probs_from_batch(model, main_kwargs, return_value = 'probs')
            batch_probs.extend(probs.reshape(-1).tolist())
    return batch_probs

def get_stat_dicts(batch, binary_correct, batch_probs, batch_preds):
    # make dicts for accuracy and probabilities, by input case type
    per_case_metadata = batch['per_case_metadata']
    unique_cases = set(item_metadata['case'] for item_metadata in per_case_metadata)
    correctness_dict = {f"{case}_acc": [] for case in unique_cases}
    seen_dict = {f"{case}_seen": [] for case in unique_cases}
    probabilities_dict = {f"{case}_prob": [] for case in unique_cases}
    outputs_dict = {f"{case}_output": [] for case in unique_cases}
    # correctness_dict
    for case_metadata, correctness in zip(per_case_metadata, binary_correct):
        case = case_metadata['case']
        correctness_dict[f"{case}_acc"].append(correctness)
    # output dict
    for case_metadata, pred in zip(per_case_metadata, batch_preds):
        case = case_metadata['case']
        outputs_dict[f"{case}_output"].append(pred)
    # seen_dict
    for case_metadata in per_case_metadata:
        case = case_metadata['case']
        seen_in_train_data = case_metadata['seen_during_training']
        seen_dict[f"{case}_seen"].append(seen_in_train_data)
    # put probability stats into dictionary (match up batched label probabilities_dict with their respective inputs)
    for case_metadata, prob in zip(per_case_metadata, batch_probs):
        case = case_metadata['case']
        stat_key = f"{case}_prob"
        probabilities_dict[stat_key].append(prob)
    return correctness_dict, seen_dict, probabilities_dict, outputs_dict

def get_prob_metrics(per_case_metadata, probabilities_dict, case_name='o_given_s_r', probs_name='pre_update_probs'):
    # compute probabilistic coherence metrics
    if f'{case_name}_prob' in probabilities_dict:
        case_targets = [case_metadata[probs_name] for case_metadata in per_case_metadata if case_metadata['case'] == case_name]
        targets = utils.combine_list_of_dicts(case_targets)
        case_probs = probabilities_dict[f'{case_name}_prob']
        # compute probabilistic coherence targets
        prob_stats = metrics.eval_probabilistic_coherence(targets, case_probs)
    else:
        prob_stats = {}
    return prob_stats

def get_logic_metrics(args, batch, probabilities_dict):
    # get A sentence T/F values if necessary
    eval_on_complex_sentences = (args.add_is_true_false_sentences or args.add_and_sentences or args.add_or_sentences or args.add_not_sentences)
    if eval_on_complex_sentences:
        if args.add_is_true_false_sentences or args.add_and_sentences or args.add_or_sentences:
            A_sentence_truth_values = [doc_dict['A_sentence']['sentence_is_true'] for doc_dict in batch['doc_dicts']]
            str_labels = batch['str_labels']
        else:
            A_sentence_truth_values = str_labels = None
        popper_stats = metrics.eval_popper_metrics(args, probabilities_dict, A_sentence_truth_values=A_sentence_truth_values, str_labels=str_labels)
    else:
        popper_stats = {}
    return popper_stats

def evaluate_model(args, 
                log,
                model, 
                dataloader, 
                tokenizer,
                model_editing_eval=False,
                verbose=False):
    '''
    main train_and_eval function that trains or evaluates models
    returns eval_stats
    '''
    # init model editing stats df
    editing_stats_df = pd.DataFrame()
    # init stats dicts. epochs_stats will be running statistics, used to compute values for eval_stats
    data_stats_df = pd.DataFrame(columns=[
        'id',
        'has_relevant_property',
        'o_given_s_r_acc',
        'A_and_B_case_acc',
        'B_and_A_case_acc',
        'A_or_B_case_acc',
        'not_A_case_acc',
        'A_TF_case_acc',
        'B_TF_case_acc',
        'o_given_s_r_seen',
        'A_and_B_case_seen',
        'B_and_A_case_seen',
        'A_or_B_case_seen',
        'not_A_case_seen',
        'A_TF_case_seen',
        'B_TF_case_seen',
        'multiplication_loss',
        'commutation_loss',
        'disjunction_loss',
        'negation_loss',
        'TF_coherence_loss',
    ])
    eval_stats = {
        'n_batches': 1,
        'forward_time_sum' : 0,
    }
    epoch_stats = {
        'o_given_s_r_loss_sum': 0,
        'seq_loss_sum': 0,
        'acc_sum': 0,
        'n_data_points': 0,
        'n_tokens': 0,
        'n_atomic_facts': 0,
        'n_sentences': 0,
        'o_given_s_r_acc_sum': 0,
        'A_and_B_case_acc_sum': 0,
        'B_and_A_case_acc_sum': 0,
        'A_or_B_case_acc_sum': 0,
        'not_A_case_acc_sum': 0,
        'A_TF_case_acc_sum': 0,
        'B_TF_case_acc_sum': 0,
        'multiplication_loss_sum': 0,
        'commutation_loss_sum': 0,
        'disjunction_loss_sum': 0,
        'negation_loss_sum': 0,
        'TF_coherence_loss_sum': 0,
    }
    logical_coherence_stats = {
        'multiplication_loss': [],
        'commutation_loss': [],
        'disjunction_loss': [],
        'negation_loss': [],
        'TF_coherence_loss': [],
    }
    probabilistic_coherence_keys = [
        'generative_prob_error',
        'data_frequency_o_given_s_r_error',
        'posterior_prob_o_given_s_r_error',
        'data_frequency_marginalized_error',
        'posterior_prob_marginalized_error',
        'data_frequency_true_conditional_error',
        'posterior_prob_true_conditional_error',
    ]
    probabilistic_coherence_stats = {k: [] for k in probabilistic_coherence_keys}
    start_time = time.time()
    model.eval()
    total_batches = len(dataloader)
    eval_on_complex_sentences = (args.add_is_true_false_sentences or args.add_and_sentences or args.add_or_sentences or args.add_not_sentences)
    print_budget = args.num_print
    for batch_num, batch in enumerate(dataloader):
        running_time = (time.time()-start_time)
        est_run_time = (running_time/(batch_num+1)*total_batches)
        forward_time = eval_stats['forward_time_sum'] / eval_stats['n_batches']
        _eval_loop_stats = {k:v for k,v in eval_stats.items() if k in ['epoch', 'loss', 'acc']}
        log.print_training_prog(_eval_loop_stats, 'EVAL', 'EVAL', batch_num, total_batches, running_time, est_run_time, forward_time)
        batch_size = len(batch['doc_dicts'])

        # unpack gen and LM batches
        gen_batch = batch['gen_batch']
        LM_batch = batch['LM_batch']

        forward_begin = time.time()
        batch_preds = get_batch_preds(args, model, tokenizer, batch)
        batch_probs = get_batch_probs(args, model, batch)
        eval_stats['forward_time_sum'] += (time.time() - forward_begin)

        # check sequence modeling loss -- this batch is a single sentence randomly sampled from the kinds of sentences that are seen during training (but they're all standalone sentences, so single-sentence docs, and no complex sentences)
        sequence_modeling_batch = batch['sequence_modeling_batch']
        main_kwargs = {
            'input_ids': sequence_modeling_batch['input_ids'],
            'attention_mask': sequence_modeling_batch['attention_mask'],
            'labels': sequence_modeling_batch['input_ids'],
        }
        utils.move_kwargs_to_gpu(main_kwargs)
        with torch.no_grad():
            log_probs = LM_utils.compute_probs_from_batch(model, main_kwargs, return_value = 'log_probs', pad_token_id=tokenizer.pad_token_id)
            n_tokens = utils.compute_num_tokens_in_batch(main_kwargs, tokenizer)
            seq_loss = -log_probs.sum() / n_tokens

        # score the generations for generative accuracy
        n_correct, binary_correct = metrics.compute_acc_sum(tokenizer, batch_preds, gen_batch['labels'], return_where_correct=True)

        # get key statistic dicts
        per_case_metadata = batch['per_case_metadata']
        correctness_dict, seen_dict, probabilities_dict, outputs_dict = get_stat_dicts(batch, binary_correct, batch_probs, batch_preds)

        # count correctness stats
        for k,v in correctness_dict.items():
            stat_key = f"{k}_sum"
            if stat_key in epoch_stats:
                epoch_stats[stat_key] += sum(v)
        
        # calculate probabilistic coherence metrics
        prob_stats = get_prob_metrics(per_case_metadata, probabilities_dict)
        for k,v in prob_stats.items():
            probabilistic_coherence_stats[k].extend(v.tolist())

        # calculate logical coherence metrics
        logic_stats = get_logic_metrics(args, batch, probabilities_dict)
        for k,v in logic_stats.items():
            logical_coherence_stats[f"{k}_loss"].extend(v)
            epoch_stats[f"{k}_loss_sum"] += v.sum()

        # EVALUATE MODEL EDITING
        if model_editing_eval:
            batch_editing_stats = evaluate_model_editing(args, model, batch, tokenizer)
            editing_stats_df = pd.concat([editing_stats_df, batch_editing_stats], ignore_index=True)
            # save as we go
            save_path = os.path.join('outputs', f'editing_stats_{args.experiment_name}.csv')
            editing_stats_df.to_csv(save_path, index=False)
            # print and save summary as we go
            editing_summary = utils.summarize_editing_stats(args, editing_stats_df)
            # print("Editing summary:")
            # print(editing_summary)
            save_path = os.path.join('outputs', f'editing_summary_{args.experiment_name}.csv')
            editing_summary.to_csv(save_path, index=False)

        # update epoch stats
        epoch_stats['n_data_points'] += batch_size
        epoch_stats['seq_loss_sum'] += seq_loss.item()
        if 'o_given_s_r_prob' in probabilities_dict:
            epoch_stats['o_given_s_r_loss_sum'] += -np.log(probabilities_dict['o_given_s_r_prob']).sum()
        # update eval stats
        for k, v in epoch_stats.items():
            if k.endswith('_sum'):
                k_avg = epoch_stats[k] / epoch_stats['n_data_points']
                eval_stats[k.replace('_sum', '')] = k_avg
        # add avg of probabilistic coherence metrics
        for k, v in probabilistic_coherence_stats.items():
            eval_stats[k] = np.nanmean(np.abs(v))
        eval_stats['o_given_s_r_loss'] = epoch_stats['o_given_s_r_loss_sum'] / epoch_stats['n_data_points']
        eval_stats['seq_loss'] = epoch_stats['seq_loss_sum'] / (batch_num+1)
        eval_stats['acc'] = epoch_stats['o_given_s_r_acc_sum'] / epoch_stats['n_data_points']
        eval_stats['n_batches'] += 1

        # collect datapoint-level stats from this batch and add to data_stats_df
        batch_stats = {
            'id': batch['ids'],
            'has_relevant_property': [doc_dict['metadata']['has_relevant_property'] for doc_dict in batch['doc_dicts']],    
        }
        batch_stats.update(correctness_dict)
        batch_stats.update(seen_dict)
        batch_stats.update(prob_stats)
        if eval_on_complex_sentences:
            batch_stats.update(logic_stats)
        batch_stats_df = pd.DataFrame(batch_stats)
        drop_acc_cols = set([item_metadata['case'] for item_metadata in per_case_metadata if item_metadata['use_for_acc_eval'] is False])
        drop_cols = [col for col in batch_stats_df.columns if any([x in col for x in drop_acc_cols])]
        batch_stats_df = batch_stats_df.drop(columns=drop_cols, axis=1)
        data_stats_df = pd.concat([data_stats_df, batch_stats_df], axis=0, ignore_index=True) if len(data_stats_df) > 0 else batch_stats_df

        # print examples
        if args.print_incorrect and print_budget > 0:
            print_idx = np.argwhere(1-binary_correct).reshape(-1).tolist()[:print_budget]
            print_budget -= len(print_idx)
        elif (batch_num == 0 and args.num_print > 0): 
            print_idx = list(range(min(args.num_print, len(batch['ids']))))
        else:
            print_idx = []
        if len(print_idx) > 0:
            print("\n" + "-"*20 + f"\nPrinting EVAL data:")
            prev_data_id = None
            for i in print_idx:
                sentence_is_true = batch['per_case_metadata'][i]['sentence_is_true']
                data_id = batch['per_case_metadata'][i]['id']
                # print logic metrics 
                if len(logic_stats) > 0 and data_id != prev_data_id:
                    print(f"Begin point {data_id}")
                    position_in_batch = batch['ids'].index(data_id)
                    print("Logic stats:", {k: round(v[position_in_batch],6) for k,v in logic_stats.items()})
                if True: # sentence_is_true:
                    model_input = tokenizer.decode(gen_batch['input_ids'][i], skip_special_tokens=False)
                    print(f" Item {data_id} | Batch input {i} | {batch['per_case_metadata'][i]['case']}")
                    print(f" Gen input(short): {tokenizer.decode(gen_batch['input_ids'][i], skip_special_tokens=True)}")
                    pred = batch_preds[i]
                    gen_label = tokenizer.decode(gen_batch['labels'][i], skip_special_tokens=False)
                    print(f" Pred        : {pred}")
                    print(f" Label       : {gen_label}")
                    lm_input = tokenizer.decode(LM_batch['input_ids'][i], skip_special_tokens=False)
                    LM_labels = tokenizer.decode(LM_batch['labels'][i], skip_special_tokens=False)
                    print(f" LM input(short): {tokenizer.decode(LM_batch['input_ids'][i], skip_special_tokens=True)}")
                    print(f" Prob           : {batch_probs[i]:.4f}")
                prev_data_id = data_id
                
            print("-"*20 + '\n')

    # plot histograms for probabilistic coherence error distributions
    prob_stats = pd.DataFrame(probabilistic_coherence_stats)
    save_name = f"probabilistic_coherence_error_{args.experiment_name}" 
    plotting_utils.plot_distributions_facet(prob_stats, save_name)

    if verbose:
        print("All stats:")
        for k,v in epoch_stats.items():
            print(f"  {k:25s}: {v:.4f}")

    return eval_stats, data_stats_df, editing_stats_df

def train_and_eval(args, 
                log, 
                model, 
                train_dataloader, 
                eval_dataloader,
                tokenizer, 
                optimizer, 
                scheduler, 
                num_train_epochs,
                training_budget_in_tokens,
                model_save_path,
                save_statistics=False):
    '''
    main train_and_eval function that trains or evaluates models
    returns train_stats and eval_stats
    '''
    # init stats dicts. epochs_stats will be running statistics, used to compute values for train_stats and eval_stats
    train_stats = {
        'n_batches': 1,
        'forward_time_sum': 0,
        'n_tokens': 0,
        'n_steps': 0,
        'n_sentences': 0
    }
    eval_stats = {} 
    total_batches = len(train_dataloader)
    start_time = time.time()
    model.train()
    eval_at_token_threshold = np.floor(training_budget_in_tokens / args.num_evals) if args.num_evals > 0 else None

    for epoch in range(1, num_train_epochs+1):
        epoch_stats = {
            'loss_sum': 0,
            'n_data_points': 0,
            'n_batches': 0,
            'n_tokens': 0,
            'n_atomic_facts': 0,
            'n_sentences': 0,
        }
        for batch_num, batch in enumerate(train_dataloader):
            running_time = (time.time()-start_time)
            est_run_time = (running_time/train_stats['n_batches']*total_batches*num_train_epochs)
            forward_time = train_stats['forward_time_sum'] / train_stats['n_batches']
            current_LR = scheduler.get_last_lr()[0] # two param groups...just take one
            batch_size = batch['input_ids'].size(0)
            log.print_training_prog(train_stats, epoch, num_train_epochs, batch_num, total_batches, 
                running_time, est_run_time, forward_time, current_LR, gpu_mem=utils.get_gpu_utilization(args.gpu))

            # forward pass on main input. if using learned optimizer and this is the before-training eval or is_update_epoch, use the original model
            main_kwargs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'labels': batch['input_ids'],
            }
            utils.move_kwargs_to_gpu(main_kwargs)
            with torch.enable_grad():
                forward_begin = time.time()
                # the default CrossEntropy loss averages over pad tokens which always have 0 loss. we manually compute the loss per sequence and average over sequences
                # main_outputs = model(**main_kwargs)
                # batch_loss = main_outputs['loss']
                log_probs = LM_utils.compute_probs_from_batch(model, main_kwargs, return_value = 'log_probs', pad_token_id=tokenizer.pad_token_id)
                n_tokens = batch['metadata']['num_tokens']
                batch_loss = -log_probs.sum() / n_tokens # avg per token
                # batch_loss = -log_probs.mean() # avg across sequences
                train_stats['forward_time_sum'] += (time.time() - forward_begin)

            # step optimizer for training a model
            loss = batch_loss / args.grad_accumulation_factor
            loss.backward()
            epoch_stats['loss_sum'] += batch_loss.item()
            epoch_stats['n_tokens'] += batch['metadata']['num_tokens']
            if (batch_num+1) % args.grad_accumulation_factor == 0 or (batch_num == total_batches-1):                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_stats['n_steps'] += 1
            del loss
            # update statistics
            epoch_stats['n_data_points'] += batch_size
            epoch_stats['n_tokens'] += batch['metadata']['num_tokens']
            # update train stats that are printed during training
            train_stats['loss'] = epoch_stats['loss_sum'] / (batch_num+1)
            train_stats['n_batches'] += 1
            train_stats['n_tokens'] += batch['metadata']['num_tokens']
            train_stats['n_sentences'] += batch['metadata']['num_sentences']

            # print first batch of first epoch
            # if batch_num == 0 and epoch == 1 and args.num_print > 0:
            sentences = [sent for sent in batch['metadata']['documents']]
            # also_print = any(['Nelson Santovenia national of' in sentence for sentence in sentences]) and batch_num < 2
            # if batch_num == 0 and (epoch == 1 or epoch % 100 == 0) and args.num_print > 0 or also_print:
            if batch_num == 0 and (epoch == 1 or epoch % 100 == 0) and args.num_print > 0:
                print("\n" + "-"*20 + f"\nPrinting TRAIN data:")
                text_data = batch['metadata']['documents']
                # print_idx = np.argwhere(['Nelson Santovenia national of' in sentence for sentence in sentences]).reshape(-1)
                # for i in print_idx:
                for i in range(min(args.num_print, batch_size)):
                    model_input = tokenizer.decode(batch['input_ids'][i])
                    print(f" Doc {i}: {text_data[i]}")
                    print(f" Model input {i}: {model_input}")
                    # print(f" Input ids   {i}: {batch['input_ids'][i].tolist()}")
                    # print(f" Attention   {i}: {(1*batch['attention_mask'][i]).tolist()}")
                    # print(f" Labels      {i}: {batch['labels'][i].tolist()}")
                print("-"*20 + '\n')

            # early breaks
            if train_stats['n_tokens'] >= training_budget_in_tokens:
                print("BREAKING MID-EPOCH BECAUSE EXCEEDING TRAINING BUDGET IN TOKENS")
                break

            # eval and log results
            # this evaluation can occur in the middle of an epoch if we are spacing out the evals via args.num_evals. always eval at end of first epoch
            # eval_at_epoch_end = (eval_at_token_threshold is None and batch_num==total_batches-1) or (epoch == 1 and batch_num==total_batches-1)
            eval_at_epoch_end = False
            token_threshold_passed = (eval_at_token_threshold is not None and train_stats['n_tokens'] >= eval_at_token_threshold)
            if eval_at_epoch_end or token_threshold_passed:
                eval_stats, eval_stats_df, editing_stats_df = evaluate_model(args,
                        log,
                        model,
                        eval_dataloader, 
                        tokenizer,
                        verbose=False)
                eval_stats_df['epoch'] = epoch
                eval_stats_df['n_tokens'] = train_stats['n_tokens']
                model.train()
                # print eval stats
                log.print_epoch_scores(epoch=epoch, scores=eval_stats)
                # make log stats
                log_stats = {
                    'LR': current_LR,
                    'epoch': epoch,
                    'step': train_stats['n_steps'],
                    'n_sentences': train_stats['n_sentences'],
                    'n_tokens': train_stats['n_tokens'],
                    'train_loss': train_stats['loss'],
                    'eval_loss': eval_stats['seq_loss'],
                    'o_given_s_r_loss': eval_stats['o_given_s_r_loss'],
                    'eval_acc': eval_stats['acc'],
                }
                log_stats.update({k:v for k,v in eval_stats.items() if '_acc' in k or '_loss' in k or '_error' in k}) # add all acc/loss entries to log_stats
                log.add_to_log(log_stats)
                group_stats = utils.summarize_eval_stats_df(eval_stats_df)
                log.add_subset_stats_to_logs(group_stats)
                # plot 
                log.save_plots()
                if token_threshold_passed:
                    eval_at_token_threshold += np.floor(training_budget_in_tokens / args.num_evals)

    # save the model!
    print(f"Saving model at {model_save_path}...")
    torch.save(model.state_dict(), model_save_path)
    return train_stats, eval_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc. & debugging
    parser.add_argument('--gpu', type = str, default = '0', help = 'comma separated list of gpu ids to use')
    parser.add_argument("--seed", default=0, type=int, help='')
    parser.add_argument("--debug", default=False, type=str2bool)
    parser.add_argument("--num_print", '-np', default = 1, type=int, help = 'number of points to print')
    parser.add_argument("--print_incorrect", default = False, type=str2bool, help = '')
    parser.add_argument("--gradient_checkpointing", default = False, type=str2bool, help = '')
    # model
    parser.add_argument("--model", default='mistral-83m', type=str, help='name of pretrained model. custom mistral sizes are 83m, 162m, 334m, and ')
    parser.add_argument("--model_class", default='clm', choices=['clm', 'clm-retrieval'], help='')
    parser.add_argument("--random_model_init", default = True, type=str2bool, help = 'use randomized model weights, NOT pretrained model weights')  
    parser.add_argument("--quantization", default = 'NA', choices=['NA', '8bit', '4bit', '16bit'], type=str, help = 'quantize model for inference')
    parser.add_argument("--padding_side", default='left', choices=['left', 'right'], help='')
    # paths/directories for data, cache, etc.
    parser.add_argument("--data_dir", default='', type=str, help='')
    parser.add_argument("--model_dir", default='', type=str, help='')
    parser.add_argument("--cache_dir", default='', type=str, help='')
    parser.add_argument("--log_dir", default='training_logs', type=str, help='')
    # training distribution parameters
    parser.add_argument("--read_n_facts", default=1e6, type=float, help='read this many facts from wikidata5m')
    parser.add_argument("--num_atomic_facts", default=1000, type=int, help='')
    parser.add_argument("--num_relations", default=None, type=int, help='')
    parser.add_argument("--min_ground_truth_prob", '-mgtp', default = 1, type=float, help = 'smooth object distributions by')
    parser.add_argument("--max_sentences_per_doc", default = 1, type=int, help = 'maximum number of sentences in a document')
    parser.add_argument("--n_base_fact_samples", '-nbfs', default = 1, type=int, help = 'number of possibly noisy samples per base fact')
    parser.add_argument("--n_complex_sentences_per_fact", '-ncs', default = 5, type=int, help = '')
    parser.add_argument("--match_n_complex_sentences_to_nbfs", default = True, type=str2bool, help = 'currently only implemented for TF complex sentences. This uses as many TF samples as there are nbfs')
    parser.add_argument("--complex_sentences_from_noisy_facts", default = False, type=str2bool, help = 'create not/and/or sentences from noisy sampled facts if mgtp<1')
    parser.add_argument("--k_homogenous_doc_order_resamples", default = 1, type=int, help = 'should be at least 1')
    parser.add_argument("--k_heterogenous_doc_order_resamples", default = 0, type=int, help = 'set to 0 to not duplicate any data')
    parser.add_argument("--add_is_true_false_sentences", default = False, type=str2bool, help = '')
    parser.add_argument("--add_or_sentences", default = False, type=str2bool, help = '')
    parser.add_argument("--add_and_sentences", default = False, type=str2bool, help = '')
    parser.add_argument("--add_if_then_sentences", default = False, type=str2bool, help = '')
    parser.add_argument("--add_not_sentences", default = False, type=str2bool, help = '')
    parser.add_argument("--complex_sentences_only", default = False, type=str2bool, help = 'this drops all "s r o" sentences from the pretraining data')
    parser.add_argument("--false_facts_use_prespecified_distractors", default = True, type=str2bool, help = 'false facts sampled as p(o|s,r) distributions use only one possible distractor object. only applies to p(o|s,r), not p(o|.,r, relevant_property)')
    parser.add_argument("--true_obj_always_most_frequent", default = True, type=str2bool, help = 'if true, regardless of the mgtp, we only accept samples of objects where the true obj is the most frequent obj for that entity')
    parser.add_argument("--max_object_tokens", default = 4, type=int, help = 'maximum number of tokens allowed in an object entity')
    parser.add_argument("--LM_filtered_ents", default = False, type=str2bool, help = 'use an LM to filter entities to high probability verbalizations')
    # training hyperparams + conditions for a task model and the optimizer
    parser.add_argument("--train_batch_size", '-tbs', default=128, type=int, help='')
    parser.add_argument("--eval_batch_size", '-ebs', default=128, type=int, help='')
    parser.add_argument("--max_seq_len", '-msl', default=-1, type=int, help='all input sequences are padded/cut to this max length if set > 0')
    parser.add_argument("--training_budget_in_tokens", "-tb", default=1e6, type=float, help='')
    parser.add_argument("--num_evals", "-ne", default=40, type=int, help='will evenly distribute this many evals during training, if > 0. o/w eval after every epoch')
    parser.add_argument("--max_grad_norm", default=1., type=float, help='')
    parser.add_argument("--grad_accumulation_factor", '-gaf', default=1, type=int, help='effective batch size = batch_size * grad_accumulation_factor')
    parser.add_argument("--lr", default=5e-5, type=float, help='')
    parser.add_argument("--weight_decay", default=1e-4, type=float, help='')
    parser.add_argument("--dropout", default=0, type=float, help='')
    parser.add_argument("--random_traintime_padding", default = True, type=str2bool, 
                        help = 'this option randomly extends the max_seq_len by [0,2*max_object_tokens], in order to eliminate overfitting to position embeddings of training inputs (improving generation acc)')
    # eval params
    parser.add_argument("--traintime_eval_size", default=1000, type=int, help='max eval size during training')
    parser.add_argument("--eval_size", '-es', default=1000, type=int, help='eval size for final eval')
    # model editing hparams
    parser.add_argument("--update_method", default='LORA', choices=['LORA'], help='')
    parser.add_argument("--update_lr", default=5e-5, type=float, help='')
    parser.add_argument("--update_steps", default=40, type=int, help='use positive int to indicate that num of steps, or -1 to update until successful')
    parser.add_argument("--update_parameters", default='all', choices=['probe', 'all', 'biases', 'MLP_down', 'MLP_all', 'attn'], help='')
    parser.add_argument("--update_layer", default=None, help='')
    parser.add_argument("--update_window_size", default=None, help='number of adjacent layers to update if specifying update_layer')
    parser.add_argument("--optimizer", default='adamw', choices=['sgd', 'adamw', 'rmsprop', 'adam'])
    parser.add_argument("--regularization", default=None, choices=['kl', 'EWC', 'all_data', None], help='')
    # belief revision eval hyperparams
    parser.add_argument("--num_random_other", default=32, type=int, help='num random other points to check perf degradation on')
    parser.add_argument("--paraphrases", default=1, type=int, help='number of paraphrases to create per point for wikidata')
    # control flow + experiment conditions
    parser.add_argument("--write_KG", default = True, type=str2bool, help = 'makes a knowledge graph')
    parser.add_argument("--write_corpus", default = False, type=str2bool, help = 'makes a pretraining corpus and eval datasets, based on the KG / BayesNet')
    parser.add_argument("--do_train", default = True, type=str2bool, help = 'finetunes a model on a task')
    parser.add_argument("--do_eval", default = True, type=str2bool, help = 'performs a full evaluation of a model')
    parser.add_argument("--do_model_editing", default = False, type=str2bool, help = 'do model editing eval')
    parser.add_argument("--update_beliefs", default = False, type=str2bool, help = 'update model beliefs')
    parser.add_argument("--save_statistics", default = False, type=str2bool, help = 'writes data-point level statistics to file')
    parser.add_argument("--overwrite_existing_data", '-oed', default = False, type=str2bool, help = 'will overwrite saved datasets such as knowledge graphs')
    parser.add_argument("--overwrite_existing_results", '-oer', default = True, type=str2bool, help = 'will overwrite past experiment even if the training log indicates a past experiment has been run')
        
    # parse + env variables + check args
    args = parser.parse_args()
    # GPU + SEED setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # import torch and all utilities that may call `import torch`
    import torch
    import utils.filter_wikidata5m as wikidata
    from data_classes.bayes_net import BayesNet, GenerativeModel
    from data_classes.dataset_generator import DatasetGenerator
    from data_classes.dataloaders import PropositionDataset
    from utils import utils, metrics, LM_utils, plotting_utils
    n_gpus = torch.cuda.device_count()
    print("GPUs available:", n_gpus)
    device = torch.cuda.device("cuda") 
    torch.cuda.set_device(device)
    # arg checks
    if args.add_not_sentences:
        assert args.add_is_true_false_sentences, "Need to add TF sentences if adding not sentences"
    
    # init experiment name, TrainingLogger, stats_dict, and saving/loading paths
    args.experiment_name = experiment_name = get_experiment_name(args)
    print(f"Starting experiment: {experiment_name} \n")
    log_file = os.path.join(args.log_dir, f"log_{experiment_name}.csv")
    log = TrainingLogger(args, log_file, experiment_name = experiment_name, overwrite_existing=args.overwrite_existing_results)
    if not os.path.exists('training_logs'):
        os.mkdir('training_logs')
    
    # check if experiment already exists
    if not (args.overwrite_existing_results or 'DEBUG' in experiment_name):
        log_written = os.path.exists(log_file)
        if log_written:
            print("Already wrote log for this experiment, and will not run again! This can be overridden with --overwrite_existing")
            sys.exit()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    pretraining_datapath = utils.get_pretraining_datapath(args)
    eval_datapath = utils.get_eval_datapath(args)

    # WRITE data if it doesn't exist or overwrite is requested
    read_n_facts = 1e6 if args.debug else (None if args.read_n_facts < 0 else args.read_n_facts)
    min_relations_per_entity = 2
    knowledge_graph_path = os.path.join(args.data_dir, f"knowledge_graph_read-{read_n_facts}.npy")
    write_KG = args.write_KG and (args.overwrite_existing_data or not os.path.exists(knowledge_graph_path))
    if write_KG:
        print("Writing data because specified arguments haven't been used yet or overwrite requested")
        print(f"Will save knowledge graph to: {knowledge_graph_path}")
        MODEL_NAME = "mistralai/Mistral-7B-v0.1"
        if args.LM_filtered_ents:
            load_in_8bit = True
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_8bit=load_in_8bit, cache_dir=args.cache_dir, low_cpu_mem_usage=True) #, torch_dtype=torch.float16)
        else:
            model = None
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, cache_dir=args.cache_dir)
        tokenizer.padding_side = args.padding_side
        knowledge_graph = wikidata.make_wikidata_KG(args,
                                args.data_dir,
                                min_relations_per_entity=min_relations_per_entity,
                                min_relation_cooccurences=1000 if not args.debug else 50,
                                top_k_relations=args.num_relations if not args.debug else 10,
                                model=model,
                                tokenizer=tokenizer,
                                batch_size=24*n_gpus,
                                read_n_facts=read_n_facts,
                                overwrite_cached_data=args.overwrite_existing_data,
                                verbose=True)
        del model, tokenizer
    else:
        start = time.time()
        print(f"Reading knowledge graph from: {knowledge_graph_path}")
        knowledge_graph = np.load(knowledge_graph_path, allow_pickle=True).item()
        print(f"(Loading took {time.time()-start:.2f} seconds)")
    # turn KG into GenerativeModel
    C_path = os.path.join(args.data_dir, f"filtered_co-occurence_read-{read_n_facts}_min-rel-{min_relations_per_entity}.npy")
    if args.write_corpus:
    # if args.overwrite_existing_data or not os.path.exists(save_path):
        print(f"Initializing generative model...")
        start = time.time()
        generative_model = GenerativeModel(args,
                                knowledge_graph, 
                                read_n_facts=read_n_facts,
                                min_relations_per_entity=min_relations_per_entity,
                                min_ground_truth_prob=args.min_ground_truth_prob,
                                relevant_relations_dict=None,
        )   

        # turn GenerativeModel into pretraining dataset
        print("Initializing dataset generator...")
        dataset_generator = DatasetGenerator(args,
                                pretraining_datapath,
                                eval_datapath,
                                generative_model, 
                                max_sentences_per_doc=args.max_sentences_per_doc, 
                                n_base_fact_samples=args.n_base_fact_samples, 
                                n_complex_sentences_per_fact=args.n_complex_sentences_per_fact, 
                                k_homogenous_doc_order_resamples=args.k_homogenous_doc_order_resamples, 
                                k_heterogenous_doc_order_resamples=args.k_heterogenous_doc_order_resamples,
                                complex_sentences_from_noisy_facts=False)

    # get data paths and write data if requested
    write_corpus = args.write_corpus # and (args.overwrite_existing_data or not os.path.exists(knowledge_graph_path))
    # assert write_corpus, "Need to equip data loading via reading to also read the list of facts to give to the BayesNet"
    if write_corpus:
        print("Writing pretraining and eval datasets...")
        dataset_generator.write_pretraining_data(n_atomic_facts=args.num_atomic_facts)
        # fit rational agent to pretraining dataset
        print("Fitting Bayesian model...")
        start = time.time()
        rational_agent = BayesNet(args,
                              dataset_generator.all_sampled_facts_with_weights,
                              knowledge_graph.all_subject_entities,
                              knowledge_graph.relations_list,
                              knowledge_graph.all_object_entities,
                              generative_model.relevant_relations_dict)
        rational_agent.fit()
        print(f"Fitting Bayesian model...took {(time.time()-start):.2f} seconds")
        # write eval dataset
        dataset_generator.write_eval_dataset(eval_size=args.eval_size, bayes_net=rational_agent, verbose=False)
    else:
        pretraining_data_stats_path = pretraining_datapath.replace('.jsonl', '_stats.npy')
        pretraining_data_stats = np.load(pretraining_data_stats_path, allow_pickle=True).item()
        for k,v in pretraining_data_stats.items():
            print(f"{k:30s}: {v}")

    # prepare for training/eval
    if args.do_train or args.do_eval:
        # load model and tokenizer
        tokenizer_name = utils.adjust_tokenizer_name(args.model) # used with custom model names
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=args.cache_dir, use_fast=True)
        tokenizer.padding_side = args.padding_side
        if args.max_seq_len > 0:
            tokenizer.max_length = args.max_seq_len
        print("Loading model...")
        model_class_dict = {'seq2seq': AutoModelForSeq2SeqLM, 'clm': AutoModelForCausalLM}
        model_class = model_class_dict[args.model_class]    
        save_load_path = utils.get_model_save_load_path(args)
        # load existing model if not training and it exists
        if os.path.exists(save_load_path) and not args.do_train:
            print(f"Loading from path: {save_load_path}")
            state_dict = torch.load(save_load_path)
            config, quantization_config = utils.get_custom_config(args)
            model = model_class.from_config(config, **quantization_config)
            model.load_state_dict(state_dict)
        # load new or pretrained model
        else:
            # load config (has some custom adjustments)
            config, quantization_config = utils.get_custom_config(args)
            if args.random_model_init:
                print("Initializing random model weights")
                model = model_class.from_config(config, **quantization_config)
            else:
                print("Loading pretrained model weights")
                model = model_class.from_pretrained(args.model, config=config, cache_dir=args.cache_dir, **quantization_config)
        # quantize model
        if args.quantization in ['NA', '16bit']:
            model = model.cuda()
        print(f"Num model parameters: {sum([np.prod(p.size()) for p in model.parameters()])/1e6:.2f}m")  
        
        # load data, optimizer, scheduler, and scaler
        print("Loading data...")
        load_start = time.time()
        dataset = PropositionDataset(args, tokenizer, train_or_eval='train')
        train_dataloader = dataset.get_train_dataloader(args, pretraining_datapath)
        eval_dataloader = dataset.get_eval_dataloader(args, eval_datapath)
        print(f"Loading data...took {round((time.time() - load_start) / 60, 2)} minutes")
        
        # subset eval_dataloader if we have more than args.max_eval_size and/or traintime_eval_size facts
        if args.eval_size > args.traintime_eval_size:
            traintime_eval_dataloader = dataset.get_eval_dataloader(args, eval_datapath, n=args.traintime_eval_size)
        else:
            traintime_eval_dataloader = eval_dataloader

    # make optimizer and scheduler after we've prepared the model. print some training optimization info
    if args.do_train or args.do_eval:
        num_tokens_in_dataset = utils.compute_num_tokens_in_dataset(train_dataloader, tokenizer)
        num_tokens_in_dataset_str = f"{num_tokens_in_dataset/1e3:.2f}k" if num_tokens_in_dataset < 1e6 else f"{num_tokens_in_dataset/1e6:.2f}m"
        num_train_epochs = int(np.ceil(args.training_budget_in_tokens / num_tokens_in_dataset))
        num_effective_batches_in_dataset = np.ceil(len(train_dataloader.dataset) / args.train_batch_size / args.grad_accumulation_factor)
        num_training_steps = num_train_epochs * num_effective_batches_in_dataset
        optimizer, scheduler = utils.load_optimizer_and_scheduler(args, model, num_training_steps)
        num_opt_params = np.sum([np.prod(params.size()) for i in range(len(optimizer.param_groups)) for params in optimizer.param_groups[i]['params']])
        print(f"Num tokens in dataset: {num_tokens_in_dataset_str}")
        print(f"Avg tokens per document: {num_tokens_in_dataset/len(train_dataloader.dataset):.2f}")
        print(f"Len train_dataloader.dataset: {len(train_dataloader.dataset)}")
        print(f"Token training budget: {int(args.training_budget_in_tokens/1e6)}m")
        print(f"Num train epochs: {num_train_epochs}")
        print(f"Train Batch size: {args.train_batch_size}")
        print(f"Effective Train Batch size: {args.train_batch_size * args.grad_accumulation_factor}")
        print(f"Num actual batches in dataset: {len(train_dataloader)}")
        print(f"Num effective batches in dataset: {num_effective_batches_in_dataset}")
        print(f"Num optimization steps: {num_training_steps}")
        # print(f"Num optimization steps: {num_training_steps/1e3:.2f}k")
        print(f"Num trainable parameters: {num_opt_params/1e6:.2f}m")
        print(f"Len traintime_eval_dataloader: {len(traintime_eval_dataloader.dataset)}")
        print(f"Len eval_dataloader: {len(eval_dataloader.dataset)}")
        args.num_train_epochs = num_train_epochs
        # adjust evaluation frequency if evaluation is sub-epoch
        # if args.num_evals > num_train_epochs:
            # args.num_evals = num_train_epochs

    # train and eval model
    start_time=time.time()
    if args.do_train:
        train_stats, eval_stats = train_and_eval(args,
                        log=log,
                        model=model,
                        train_dataloader=train_dataloader, 
                        eval_dataloader=traintime_eval_dataloader,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        num_train_epochs=num_train_epochs, # defined from training budget given in terms of tokens
                        training_budget_in_tokens=args.training_budget_in_tokens,
                        model_save_path=save_load_path,
        )
        time_msg = utils.format_time(start=start_time, end=time.time())
        print(f"Training took", time_msg)

    # final model evaluation
    if args.do_eval:
        print(f"Beginning final eval for {experiment_name}...")
        eval_time = time.time()
        eval_stats, eval_stats_df, editing_stats_df = evaluate_model(args,
                log=log,
                model=model, 
                dataloader=eval_dataloader, 
                tokenizer=tokenizer,
                model_editing_eval=args.do_model_editing,
                verbose=False)
        eval_time_msg = utils.format_time(start=eval_time, end=time.time())

        train_loss = train_stats['loss'] if args.do_train else -1
        final_msg = f"train loss: {train_loss:.2f} | eval acc: {eval_stats['acc']:.2f}"
        print("\nFinal results (aggregated):")
        log.print_epoch_scores(epoch = 'EVAL', scores = eval_stats)
        group_stats = utils.summarize_eval_stats_df(eval_stats_df, verbose=False)
        # save eval stats df
        save_path = os.path.join('outputs', f'eval_stats_{args.experiment_name}.csv')
        eval_stats_df.to_csv(save_path, index=False)
        save_path = os.path.join('outputs', f'eval_summary_{args.experiment_name}.csv')
        one_row_summary = pd.DataFrame([group_stats['all']])
        one_row_summary.to_csv(save_path, index=False)
        if args.do_model_editing:
            save_path = os.path.join('outputs', f'editing_stats_{args.experiment_name}.csv')
            editing_stats_df.to_csv(save_path, index=False)
            editing_summary = utils.summarize_editing_stats(args, editing_stats_df)
            print("Editing summary:")
            print(editing_summary)
            save_path = os.path.join('outputs', f'editing_summary_{args.experiment_name}.csv')
            editing_summary.to_csv(save_path, index=False)
            
        print()
        print(final_msg)
        print("Eval time: ", eval_time_msg)
        # log stats
        if args.do_train:
            print("Training time: ", time_msg)
            log_stats = {
                'LR': -1,
                'epoch': num_train_epochs,
                'step': train_stats['n_steps'],
                'n_sentences': train_stats['n_sentences'],
                'n_tokens': train_stats['n_tokens'],
                'train_loss': train_stats['loss'],
                'eval_loss': eval_stats['seq_loss'],
                'o_given_s_r_loss': eval_stats['o_given_s_r_loss'],
                'eval_acc': eval_stats['acc'],
            }
            log_stats.update({k:v for k,v in eval_stats.items() if '_acc' in k or '_loss' in k or '_error' in k}) # add all acc/loss entries to log_stats
            log.add_to_log(log_stats)
            log.save_plots()

            # update the one row summary to include train stats
            save_path = os.path.join('outputs', f'eval_summary_{args.experiment_name}.csv')
            one_row_summary = pd.DataFrame([log_stats])
            one_row_summary.to_csv(save_path, index=False)