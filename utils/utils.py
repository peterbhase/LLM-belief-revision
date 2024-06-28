import argparse
import numpy as np
import pandas as pd
import os
import pynvml
import re
import sys

from transformers import AutoConfig

def small_diff(x, target):
    return np.abs(x - target) < 1e-4

def dict_to_df(dict_object):
    stats_row_df = pd.DataFrame(dict_object, index=[0])
    bool_cols = stats_row_df.select_dtypes(include='bool').columns
    stats_row_df[bool_cols] = stats_row_df[bool_cols].astype(bool)
    return stats_row_df

def cast_bool_df(df):
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(bool)
    return df

def human_readable_object_size(object):
    size = sys.getsizeof(object)
    size /= 1024
    if size < 1024:
        return f"{int(size):4d} MB."
    if size >= 1024:
        size /= 1024
        return f"{int(size):4d} GB."

def compute_num_tokens_in_batch(batch, tokenizer):
    input_ids = batch['input_ids']
    non_special_tokens = (input_ids != tokenizer.pad_token_id) * (input_ids != tokenizer.eos_token_id) * (input_ids != tokenizer.bos_token_id)
    attended_non_special_tokens = non_special_tokens * batch['attention_mask']
    num_tokens = attended_non_special_tokens.sum()
    return num_tokens.item()

def summarize_eval_stats_df(eval_stats_df, verbose=False):
    group_name_to_results = {}
    avg_metrics = [k for k in eval_stats_df.columns if 'acc' in k or 'loss' in k or 'error' in k or 'epoch' in k or 'tokens' in k]
    drop_cols = ['A_sentence_acc', 'not_A_with_A_TF_label_case_acc']
    avg_metrics = [metric for metric in avg_metrics if metric not in drop_cols]
    # take abs value of probabilistic error metrics
    for metric in [metric for metric in avg_metrics if 'error' in metric]:
        eval_stats_df[metric] = np.abs(eval_stats_df[metric])
    # first get avg of the whole df
    summary_df = eval_stats_df[avg_metrics].mean(skipna=True)
    group_name_to_results["all"] = summary_df
    # now get summaries for breakdowns by these grouping variables
    grouping_vars = ['has_relevant_property'] # do not put 'x_seen' variables here
    for grouping_var in grouping_vars:
        summary_df = average_df_over_metrics(eval_stats_df, [grouping_var], avg_metrics)
        if verbose:
            print(grouping_var, " <-- grouping by this variable")
            printable = summary_df.T.reset_index().sort_values(by='index', key=lambda x : x.str[::-1])
            print(printable) 
        group_values = list(set(summary_df[grouping_var]))
        for group_value in group_values:
            group_name = f"{grouping_var}={group_value}"
            subset_rows = summary_df[summary_df[grouping_var]==group_value]
            group_name_to_results[group_name] = subset_rows #.to_dict('records')[0]
    # now summarize avg metrics for seen/unseen sentences
    # first do the o_given_s_r_seen variable, which includes more metrics
    if 'o_given_s_r_seen' in eval_stats_df.columns:
        o_given_s_r_metrics = [metric for metric in avg_metrics if 'o_given_s_r' in metric or 'loss' in metric or 'error' in metric or 'epoch' in metric or 'tokens' in metric]
        agg_df = eval_stats_df.groupby('o_given_s_r_seen', as_index=False).agg({**{metric: 'mean' for metric in o_given_s_r_metrics}})
        agg_df['n'] = eval_stats_df.groupby('o_given_s_r_seen', as_index=False).size()['size']
        required_values_df = pd.DataFrame({'o_given_s_r_seen': [True, False]})
        avg_df = pd.merge(required_values_df, agg_df, on='o_given_s_r_seen', how='left')
        avg_df['n'].fillna(0, inplace=True)
        avg_df['n'] = avg_df['n'].astype(int)
        avg_df = avg_df.sort_values(by='o_given_s_r_seen')
        seen_dfs = [avg_df]
    else:
        seen_dfs = []
    # now add seen_dfs for other individual variables
    grouping_stats = [k for k in avg_metrics if ('acc' in k) and not 'o_given_s_r' in k]
    # grouping_stats = [k for k in avg_metrics if ('acc' in k or 'epoch' in k) and not 'o_given_s_r' in k]
    grouping_vars = [k.replace('acc', 'seen') for k in grouping_stats]
    for grouping_var, grouping_stat in zip(grouping_vars, grouping_stats):
        subset_df = eval_stats_df.groupby(grouping_var, as_index=False).agg({grouping_stat: 'mean'})
        n_name = grouping_var.replace("seen", "n")
        subset_df[n_name] = eval_stats_df.groupby(grouping_var, as_index=False).size()['size']
        required_values_df = pd.DataFrame({grouping_var: [True, False]})
        subset_df = pd.merge(required_values_df, subset_df, on=grouping_var, how='left')
        subset_df[n_name].fillna(0, inplace=True)
        subset_df[n_name] = subset_df[n_name].astype(int)
        subset_df = subset_df.sort_values(by=grouping_var)
        seen_dfs.append(subset_df)
    seen_df = pd.concat(seen_dfs, axis=1)
    seen_df['seen'] = [False, True]
    seen_df.fillna(0, inplace=True)
    for group_value in [False, True]:
        group_name = f"seen={group_value}"
        subset_rows = seen_df[seen_df['seen']==group_value]
        group_name_to_results[group_name] = subset_rows
    seen_df = seen_df[sorted(seen_df.columns)]
    group_name_to_results['seen'] = seen_df
    if verbose:
        print("seen", " <-- grouping by this variable")
        keep_cols = [col for col in seen_df.columns if col == 'seen' or not 'seen' in col]
        seen_df = seen_df[keep_cols]
        printable = seen_df.T.reset_index().sort_values(by='index', key=lambda x : x.str[::-1])
        print(printable)
    return group_name_to_results

def summarize_editing_stats(args, editing_stats_df):
    init_stats = editing_stats_df[editing_stats_df.step == 0]
    final_stats = editing_stats_df[editing_stats_df.step == args.update_steps]
    acc_cols = [col for col in init_stats.columns if 'o_given_s_r_acc' in col or 'same_s_same_r_fact_requested_obj_acc' in col or 'new_modal_obj_acc' in col or 'GT_obj_acc' in col]
    logic_cols = ['TF_coherence', 'negation', 'disjunction', 'multiplication', 'commutation']
    summarize_cols = acc_cols + [x for x in logic_cols if x in editing_stats_df.columns]
    init_summary = init_stats[summarize_cols].mean().reset_index()
    final_summary = final_stats[summarize_cols].mean().reset_index()
    init_summary = init_summary.sort_values(by='index', key = lambda x: list(reversed(x)))
    final_summary = final_summary.sort_values(by='index', key = lambda x: list(reversed(x)))
    init_summary.columns = ["metric", "pre-edit"]
    final_summary.columns = ["metric", "post-edit"]
    all_summary = pd.concat([init_summary, final_summary['post-edit']], axis=1)
    all_summary = all_summary.sort_values(by='metric', key = lambda x: list(reversed(x)))
    return all_summary

def average_df_over_metrics(df, grouping_vars, metrics_vars):
    # averages the metrics_vars columns in a df, while keeping grouping_vars
    collapsed_dfs = []
    for metric in metrics_vars:
        if metric in df.columns:
            avg_df = df.groupby(grouping_vars)[metric].mean().reset_index()
            try:
                avg_df['n'] = int(df.groupby(grouping_vars)['id'].count())
            except:
                avg_df['n'] = df.groupby(grouping_vars)['id'].count()
            collapsed_dfs.append(avg_df)
    joined_df = collapsed_dfs[0]
    for collapsed_df in collapsed_dfs[1:]:
        joined_df = joined_df.merge(collapsed_df)
    return joined_df

def flip_probs_to_prob_true(probs_stats, str_labels):
    prob_true_stats = {}
    # The first thing we do is flip the probabilities for inputs like "X is False", so that we are always looking at p(True). This ASSUMES that p(True|s r o is) = 1-p(False|s r o is), which should be trivial for the model to learn
    for case, probs in probs_stats.items():
        has_TF_labels = case in ['A_or_B_case_prob', 'not_A_case_prob', 'A_TF_case_prob', 'B_TF_case_prob', 'A_and_B_case_prob', 'B_and_A_case_prob']
        if has_TF_labels:
            true_false_labels = str_labels[case.replace('_prob', '')]
            prob_true_stats[case] = np.array([prob if label=='true' else 1-prob for prob, label in zip(probs, true_false_labels)])
        else:
            prob_true_stats[case] = probs
    return prob_true_stats

def load_optimizer_and_scheduler(args, model, num_training_steps):
    from torch.optim import AdamW, SGD, RMSprop, Adam
    from transformers import get_scheduler
    from transformers.optimization import Adafactor, AdafactorSchedule
    named_parameters = model.named_parameters()
    optimizer_grouped_parameters = [
        {"params": [p for n, p in named_parameters if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": args.weight_decay,
            'lr' : args.lr},
        {"params": [p for n, p in named_parameters if any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": 0.0,
            'lr' : args.lr}
    ]
    optimizer_to_class = {'adamw' : AdamW, 'sgd' : SGD, 'rmsprop' : RMSprop, 'adam' : Adam, 'adafactor': Adafactor}
    optimizer_class = optimizer_to_class[args.optimizer]
    optimizer = optimizer_class(optimizer_grouped_parameters)
    # scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    if args.optimizer != "adafactor":
        percent_of_orig_value = .1 # drop down to this percent of original LR by end of training
        multiplier = 1 / (1-percent_of_orig_value)
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=multiplier*num_training_steps)
    else:
        scheduler = AdafactorSchedule(optimizer)
    return (optimizer, scheduler)

def load_update_optimizer(args, model):
    from torch.optim import AdamW
    from transformers import get_scheduler
    named_parameters = model.named_parameters()
    optimizer_grouped_parameters = [
        {"params": [p for n, p in named_parameters if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": args.weight_decay,
            'lr' : args.update_lr},
        {"params": [p for n, p in named_parameters if any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": 0.0,
            'lr' : args.update_lr}
    ]
    optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=2e8)
    return (optimizer, scheduler)

def get_dataset_identifier(args):
    # define complex sentences insert
    complex_sentences_insert = []
    if args.add_is_true_false_sentences:
        complex_sentences_insert.append("TF")
    if args.add_and_sentences:
        complex_sentences_insert.append("a")
    if args.add_or_sentences:
        complex_sentences_insert.append("o")
    if args.add_not_sentences:
        complex_sentences_insert.append("n")
    if args.add_if_then_sentences:
        complex_sentences_insert.append("t")
    if args.complex_sentences_only:
        complex_sentences_insert.append("only")
    if args.complex_sentences_from_noisy_facts:
        complex_sentences_insert.append("noisy")
    if len(complex_sentences_insert) > 0:
        complex_sentences_insert = "_" + "-".join(complex_sentences_insert)
    else:
        complex_sentences_insert = ""
    return f"data_n-{args.num_atomic_facts}_r-{args.num_relations}_s-{args.max_sentences_per_doc}_b-{args.n_base_fact_samples}_mgtp-{args.min_ground_truth_prob}_c-{args.n_complex_sentences_per_fact}{complex_sentences_insert}_ho-{args.k_homogenous_doc_order_resamples}_he-{args.k_heterogenous_doc_order_resamples}"

def get_pretraining_datapath(args):
    dataset_identifier = get_dataset_identifier(args)
    pretraining_dataname = f"pretraining_{dataset_identifier}.jsonl"
    pretraining_datapath = os.path.join(args.data_dir, pretraining_dataname)
    return pretraining_datapath

def get_eval_datapath(args):
    dataset_identifier = get_dataset_identifier(args)
    eval_dataname = f"eval_{dataset_identifier}_es-{args.eval_size}.jsonl"
    eval_datapath = os.path.join(args.data_dir, eval_dataname)
    return eval_datapath

def get_experiment_name(args):
    tb = f"{args.training_budget_in_tokens:.1e}"
    effective_batch_size = args.train_batch_size * args.grad_accumulation_factor
    # define complex sentences insert
    complex_sentences_insert = []
    if args.add_is_true_false_sentences:
        complex_sentences_insert.append("TF")
    if args.add_and_sentences:
        complex_sentences_insert.append("a")
    if args.add_or_sentences:
        complex_sentences_insert.append("o")
    if args.add_not_sentences:
        complex_sentences_insert.append("n")
    if args.add_if_then_sentences:
        complex_sentences_insert.append("t")
    if args.complex_sentences_only:
        complex_sentences_insert.append("only")
    if args.complex_sentences_from_noisy_facts:
        complex_sentences_insert.append("noisy")
    if len(complex_sentences_insert) > 0:
        complex_sentences_insert = "_" + "-".join(complex_sentences_insert)
    else:
        complex_sentences_insert = ""
    # make experiment name
    short_model = args.model.split('/')[-1]
    if args.model == "openai-community/gpt2":
        short_model += '-small'
    experiment_name = f"{short_model}_{args.model_class}_{args.num_atomic_facts}-facts_{args.num_relations}-rels_tb-{tb}_bs-{effective_batch_size}_mgtp-{args.min_ground_truth_prob:.1f}" + \
        f"_s-{args.max_sentences_per_doc}_b-{args.n_base_fact_samples}_c-{args.n_complex_sentences_per_fact}{complex_sentences_insert}_ho-{args.k_homogenous_doc_order_resamples}_he-{args.k_heterogenous_doc_order_resamples}" + \
        f"_sd{args.seed}"
    experiment_name = experiment_name.replace('/', '-')
    if args.debug:
        experiment_name += "_DEBUG"
    return experiment_name

def combine_list_of_dicts(list_of_dicts):
    all_keys = set([key for _dict in list_of_dicts for key in _dict.keys()])
    return_dict = {key: [] for key in all_keys}
    for _dict in list_of_dicts:
        for k,v in _dict.items():
            return_dict[k].append(v)
    return return_dict

def get_gpu_utilization(gpu=0):
    if len(gpu) > 1:
        gpu = gpu.split(',')[0]
    gpu = int(gpu)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return f"{info.used//1024**2} MB."

def adjust_tokenizer_name(model):
    if model == 'facebook/opt-5b':
        tokenizer_name = 'facebook/opt-350m'
    elif model == 'gpt2-3b' or model == 'gpt2-5b':
        tokenizer_name = 'gpt2-medium'
    elif 'mistral' in model:
        tokenizer_name = "mistralai/Mistral-7B-v0.1"
    else:
        tokenizer_name = model
    return tokenizer_name

def get_custom_config(args):
    # define variables for custom model configs
    from torch import float16
    custom_models = ['mistral-' + x for x in ['83m', '162m', '334m', '1b']]
    use_custom_model = args.model in custom_models
    if use_custom_model:
        config = AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1", cache_dir=args.cache_dir)
        if '83m' in args.model:
            config.hidden_size = 512
            config.intermediate_size = 512*4
            config.num_attention_heads = 8
            config.num_hidden_layers = 12
        if '162m' in args.model:
            config.hidden_size = 768
            config.intermediate_size = 768*4
            config.num_attention_heads = 8
            config.num_hidden_layers = 12
        if '334m' in args.model:
            config.hidden_size = 1024
            config.intermediate_size = 1024*4
            config.num_attention_heads = 8
            config.num_hidden_layers = 16
        if '1b' in args.model:
            config.hidden_size = 1520
            config.intermediate_size = 1520*4
            config.num_attention_heads = 8
            config.num_hidden_layers = 24
        config._name_or_path = args.model
    elif not use_custom_model:
        config = AutoConfig.from_pretrained(args.model, cache_dir=args.cache_dir)
    # edit config for dropout
    if args.dropout >= 0:
        allowed_models = ['facebook/opt', 'gpt2', 'llama', 'mistral']
        assert any([x in args.model for x in allowed_models]), f"if overriding dropout during training, need to use model in {allowed_models} or extend this in utils.get_custom_config"
        for k,v in config.__dict__.items():
            # this is for gpt2 and opt models
            if 'pdrop' in k or 'dropout' in k or 'attn_drop' in k:
                setattr(config, k, args.dropout)
    # quantization
    quantization_config = {}
    load_8bit = args.quantization == '8bit'
    load_4bit = args.quantization == '4bit'
    if not any([x in args.model.lower() for x in ['gpt2', 'mistral']]):
        quantization_config['load_in_4bit'] = load_4bit
        quantization_config['load_in_8bit'] = load_8bit
    if args.quantization == '16bit':
        quantization_config['torch_dtype'] = float16
    return config, quantization_config

def slice_batch_kwargs(batch_kwargs, idx):
    return_dict = {}
    for k,v in batch_kwargs.items():
        if v is None:
            return_dict[k] = v
        elif type(v) is list:
            return_dict[k] = [v[i] for i in idx]
        else:
            return_dict[k] = v[idx,...]
    return return_dict

def subset_batch_by_data_ids(batch, ids):
    per_case_indexing = ['gen_batch', 'LM_batch', 'per_case_metadata']
    per_item_indexing = ['per_item_metadata', 'sequence_modeling_batch', 'doc_dicts', 'ids']
    new_batch = {}
    for k, v in batch.items():
        if k in per_case_indexing:
            cases_for_this_item_id = np.argwhere([metadata['id'] in ids for metadata in batch['per_case_metadata']]).reshape(-1)
        elif k in per_item_indexing:
            cases_for_this_item_id = np.argwhere([metadata['id'] in ids for metadata in batch['per_item_metadata']]).reshape(-1)
        if type(v) is dict:
            new_batch[k] = slice_batch_kwargs(v, cases_for_this_item_id)
        elif type(v) is list:
            new_batch[k] = [v[i] for i in cases_for_this_item_id]
    return new_batch

def PEFT_wrap_model(args, model):
    from peft import get_peft_model, LoraConfig, TaskType
    task_type = TaskType.CAUSAL_LM
    peft_config = LoraConfig(
        task_type=task_type, inference_mode=False, r=1, lora_alpha=8, lora_dropout=0
    )
    if 'mistral' in args.model.lower() or 'mixtral' in args.model.lower():
        peft_config.target_modules = ["down_proj"]
    assert any([x in args.model.lower() for x in ['llama', 'gpt-j', 'mistral', 'persimmon', 'mpt', 'falcon', 'qwen']]), f"\nNeed to add QLoRA params to peft_config manually -- add exact q_proj and v_proj layer paths to peft_config.target_modules = [paths] from the model: \n{model} \n(SEE MESSAGE ABOVE)"
    model = get_peft_model(model, peft_config)
    return model

def str2bool(v):
    # used for boolean argparse values
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def chunk_array(array, size):
    # return chunks from the array of size=size, in left to right order
    # if array % size != 0, then last components of the array are also added, but will not be of size=size
    if len(array) <= size:
        return [array]
    start_idx = 0
    chunks = []
    for end_idx in range(1, len(array)+1):
        if end_idx % size == 0 or end_idx == len(array):
            chunks.append(array[start_idx:end_idx])
            start_idx = end_idx
    return chunks

def min_max_mean(array):
    return {'min': round(np.min(array),2), 'mean:': round(np.mean(array),2), 'max:': round(np.max(array),2)}

def move_kwargs_to_gpu(kwargs):
    from torch import Tensor
    for k,v in kwargs.items():
        if type(v) is Tensor:
            kwargs[k] = v.cuda(non_blocking=True)

def get_model_save_load_path(args):
    experiment = get_experiment_name(args)
    model_path = os.path.join(args.model_dir, f"{experiment}.pt")
    return model_path

def format_large_number(number):
    if number < 1e6:
        return f"{(number/1e3):.2f}k"
    elif number < 1e9:
        return f"{(number/1e6):.2f}m"
    else:
        return f"{(number/1e9):.2f}b"

def format_time(start, end):
    time_diff = (end-start) / 60
    unit = 'minutes' if time_diff < 60 else 'hours'
    time_diff = time_diff if time_diff < 60 else time_diff / 60
    time_msg = f"{time_diff:.2f} {unit}"
    return time_msg

def compute_num_tokens_in_dataset(dataloader, tokenizer):
    # assumes that tokenizer has bos and eos tokens that are used in tokenization
    num_tokens = 0
    for batch in dataloader:
        input_ids = batch['input_ids']
        num_tokens += (input_ids != tokenizer.pad_token_id).sum()
        # non_special_tokens = (input_ids != tokenizer.pad_token_id) * (input_ids != tokenizer.eos_token_id) * (input_ids != tokenizer.bos_token_id)
        # attended_non_special_tokens = non_special_tokens * batch['attention_mask']
        # num_tokens += attended_non_special_tokens.sum()
    return num_tokens.item()

def str2arg(v):
    if v.lower() in ('yes', 'true', 't', 'y') + ('no', 'false', 'f', 'n'):
        return str2bool(v)
    elif v.lower()[0] == '.' or v.lower()[:1] == '0.':
        try:
            return float(v)
        except:
            pass
    else:
        try:
            return int(float(v))
        except:
            pass
    return v

def args_from_cli_command(command):
    class DummyArgs:
        pass
    dummy_args = DummyArgs()
    command_dict = {}
    items = command.split()
    for idx, item in enumerate(items):
        if idx == len(items)-1:
            break
        if item[:2] == '--':
            k = item[2:]
            v = items[idx+1]
            command_dict[k] = str2arg(v)
        elif item[0] == '-':
            k = item[1:]
            v = items[idx+1]
            command_dict[k] = str2arg(v)
    for k,v in command_dict.items():
        setattr(dummy_args, k, v)
    return dummy_args

def get_experiment_name_from_command(command):
    args = args_from_cli_command(command)
    experiment_name = get_experiment_name(args)
    return experiment_name
