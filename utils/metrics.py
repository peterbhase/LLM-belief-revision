import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def eval_probabilistic_coherence(targets_dict, probabilities):
    """
    We get the probability computed by the model for object o given prompt "s r ___"
    This could be compared against a few targets (see dataset_generator.write_eval_dataset), which should be provided in targets

    args:
        targets_dict: dict of possible targets for the probability. maps from case to list of targets
        probability: probability of object o given prompt "s r ____"
    returns:
        dict of losses, which are abs diff between probability and target for each target
    """
    return_dict = {}
    probabilities = np.array(probabilities)
    keys = [
        'generative_prob',
        'data_frequency_o_given_s_r',
        'posterior_prob_o_given_s_r',
        'data_frequency_marginalized',
        'posterior_prob_marginalized',
        'data_frequency_true_conditional',
        'posterior_prob_true_conditional',
    ]
    for key in keys:
        target = np.array(targets_dict[key])
        error = probabilities - target
        return_dict[key + "_error"] = error
    return return_dict

def eval_popper_metrics(args, probs_stats, A_sentence_truth_values, str_labels):
    """
    Popper metrics
        0. P(A is true) = P(A)
        1. P(A and B) = p(A) * P(B)
        2. P(A and B) = p(B and A)
        3. P(A or B) = p(A) + p(B) - p(A and B)
        4. P(not A) = 1 - p(A)
    args:
        probs_stats: dict of stats corresponding to types of sentences
    returns:
        dict of metrics for Popper metrics
    """
    return_dict = {}
    probs_stats = {k: np.array(v) for k,v in probs_stats.items()}
    prob_true_stats = {}
    # The first thing we do is flip the probabilities for inputs like "X is False", so that we are always looking at p(True). This ASSUMES that p(True|s r o is) = 1-p(False|s r o is), which should be trivial for the model to learn
    for case, probs in probs_stats.items():
        has_TF_labels = case in ['A_or_B_case_prob', 'not_A_case_prob', 'A_TF_case_prob', 'B_TF_case_prob', 'A_and_B_case_prob', 'B_and_A_case_prob']
        if has_TF_labels:
            true_false_labels = str_labels[case.replace('_prob', '')]
            prob_true_stats[case] = np.array([prob if label=='true' else 1-prob for prob, label in zip(probs, true_false_labels)])
    # Now compute logical metrics
    if args.add_is_true_false_sentences:
        A_TF_case_prob = probs_stats['A_TF_case_prob']
        A_sentence_prob = probs_stats['A_sentence_prob']
        '''
        If A = s r o
        - A_TF = s r o is True
        - p(o|s,r) = p(True|s r o is) # we implement this
        If A = s r o’
        - A_TF = s r o’ is False
        - p(o’|s,r) = p(True|s r o’ is)
        - p(o’|s,r) = 1 - p(False|s r o’ is) # we implement this
        Should we also check that p(True|s r o is) = 1-p(False|s r o is)? No, that should be trivial. But our implementation relies on this holding
        '''
        targets = np.array([_A_TF_case_prob if A_True else 1-_A_TF_case_prob for _A_TF_case_prob, A_True in zip(A_TF_case_prob, A_sentence_truth_values)])
        metric_five = np.abs(A_sentence_prob - targets)
        return_dict['TF_coherence'] = metric_five
    if args.add_and_sentences:
        '''
        Want to compute p(A and B) = p(A) * p(B)
        - therefore need the T/F labels on the A_and_B, A, and B inputs to match up. We simulate this by flipping all the probabilities to p(True) above
        '''
        A_and_B_case_prob = prob_true_stats['A_and_B_case_prob']
        B_and_A_case_prob = prob_true_stats['B_and_A_case_prob']
        B_TF_case_prob = prob_true_stats['B_TF_case_prob']
        metric_one = np.abs(A_and_B_case_prob - A_TF_case_prob * B_TF_case_prob)
        metric_two = np.abs(B_and_A_case_prob - A_and_B_case_prob)
        return_dict['multiplication'] = metric_one
        return_dict['commutation'] = metric_two
    if args.add_or_sentences:
        '''
        Want to compute p(A or B) = p(A) + p(B) - p(A)*p(B)
        - therefore need the T/F labels on the A_and_B, A, and B inputs to match up. We simulate this by flipping all the probabilities to p(True) above
        '''
        A_or_B_case_prob = prob_true_stats['A_or_B_case_prob']
        A_TF_case_prob = prob_true_stats['A_TF_case_prob']
        B_TF_case_prob = prob_true_stats['B_TF_case_prob']
        quantity_one = A_or_B_case_prob
        quantity_two = A_TF_case_prob + B_TF_case_prob - A_TF_case_prob*B_TF_case_prob
        metric_three = np.abs(quantity_two - quantity_one)
        return_dict['disjunction'] = metric_three
    if args.add_not_sentences:
        '''
        If A = "s r o" is True
        - Not_A = "not s r o" is True
        - p(True|s,r,o) = 1 - p(True|not s r o is) # implement this
        If A = s r o’ is False
        - Not_A = not s r o’ is False
        - p(False|s,r,o) = 1 - p(False|not s r o is) # implement this, but recall we have flipped all the probabilities to p(True)
        '''
        A_prob = prob_true_stats['A_TF_case_prob']
        not_A_case_prob = prob_true_stats['not_A_case_prob']
        one_minus_not_A = 1 - not_A_case_prob
        metric_four = np.abs(A_prob - one_minus_not_A)
        return_dict['negation'] = metric_four
    return return_dict

# def eval_popper_metrics(args, probs_stats, A_sentence_truth_values):
#     """
#     Popper metrics
#         0. P(A is true) = P(A)
#         1. P(A and B) = p(A) * P(B)
#         2. P(A and B) = p(B and A)
#         3. P(A or B) = p(A) + p(B) - p(A and B)
#         4. P(not A) = 1 - p(A)
#     args:
#         probs_stats: dict of stats corresponding to types of sentences
#     returns:
#         dict of metrics for Popper metrics
#     """
#     return_dict = {}
#     probs_stats = {k: np.array(v) for k,v in probs_stats.items()}
#     if args.add_is_true_false_sentences:
#         A_TF_case_prob = probs_stats['A_TF_case_prob']
#         A_sentence_prob = probs_stats['A_sentence_prob']
#         '''
#         If A = s r o
#         - A_TF = s r o is True
#         - p(o|s,r) = p(True|s r o is) # we implement this
#         If A = s r o’
#         - A_TF = s r o’ is False
#         - p(o’|s,r) = p(True|s r o’ is)
#         - p(o’|s,r) = 1 - p(False|s r o’ is) # we implement this
#         Should we also check that p(True|s r o is) = 1-p(False|s r o is)? No, that should be trivial. But our implementation relies on this holding
#         '''
#         targets = np.array([_A_TF_case_prob if A_True else 1-_A_TF_case_prob for _A_TF_case_prob, A_True in zip(A_TF_case_prob, A_sentence_truth_values)])
#         metric_five = np.abs(A_sentence_prob - targets)
#         return_dict['TF_coherence'] = metric_five
#     if args.add_and_sentences:
#         '''
#         Want to compute p(A and B) = p(A) * p(B)
#         - therefore need the T/F labels on the A_and_B, A, and B inputs to match up
#         - if labels are T, can use model probs
#         - if labels are F, should flip probs to 1-prob. THIS assumes p(True|s r o is) = 1-p(False|s r o is), but this is trivial for the model to meet
#         '''
#         A_and_B_case_prob = probs_stats['A_and_B_case_prob']
#         B_and_A_case_prob = probs_stats['B_and_A_case_prob']
#         B_TF_case_prob = probs_stats['B_TF_case_prob']
#         metric_one = np.abs(A_and_B_case_prob - A_TF_case_prob * B_TF_case_prob)
#         metric_two = np.abs(B_and_A_case_prob - A_and_B_case_prob)
#         return_dict['multiplication'] = metric_one
#         return_dict['commutation'] = metric_two
#     if args.add_or_sentences:
#         '''
#         Want to compute p(A or B) = p(A) + p(B) - p(A*B)
#         - therefore need the T/F labels on the A_and_B, A, and B inputs to match up
#         - if labels are T, can use model probs
#         - if labels are F, should flip probs to 1-prob. THIS assumes p(True|s r o is) = 1-p(False|s r o is), but this is trivial for the model to meet
#         '''
#         A_and_B_case_prob = probs_stats['A_and_B_case_prob']
#         A_or_B_case_prob = probs_stats['A_or_B_case_prob']
#         B_TF_case_prob = probs_stats['B_TF_case_prob']
#         quantity_one = A_or_B_case_prob
#         quantity_two = A_TF_case_prob + B_TF_case_prob - A_and_B_case_prob
#         metric_three = np.abs(quantity_two - quantity_one)
#         return_dict['disjunction'] = metric_three
#     if args.add_not_sentences:
#         '''
#         If A = "s r o" is True
#         - Not_A = "not s r o" is True
#         - p(True|s,r,o) = 1 - p(True|not s r o is)
#         If A = s r o’ is False
#         - Not_A = not s r o’ is False
#         - p(False|s,r,o) = 1 - p(False|not s r o is)
#         So we have A_TF sentence and not_A_with_A_TF which uses the same T/F label as A_TF (i.e., this is Not_A above)
#         In both cases, the metric is the same
#         '''
#         A_prob = probs_stats['A_TF_case_prob']
#         not_A_case_prob = probs_stats['not_A_with_A_TF_label_case_prob']
#         one_minus_not_A = 1 - not_A_case_prob
#         metric_four = np.abs(A_prob - one_minus_not_A)
#         return_dict['negation'] = metric_four
#     return return_dict

def force_not_dimensionless(data):
    if type(data) is torch.Tensor:
        if data.dim()==0:
            data = data.view(1)
    return data

def safe_seq(seq):
    # filter to non -100 values in seq, which is a list. -100 is the default ignore_index in pytorch
    return [x for x in seq if x >= 0]

def standardize_preds_or_labels(data, tokenizer):
    """
    takes tensors, arrays, and lists, and returns standardized pred/label strs
    IF there are multiple labels per item, then we return a list of lists
    ELSE, we return an np array
    args:
        data: should be 1-d np.array, 1-d torch.tensor, or list of these things
        tokenizer: model tokenizer
    """
    # unravel data into list or list of lists
    if type(data) is list and type(data[0]) is torch.Tensor or type(data[0]) is np.ndarray:
        data = [item.tolist() for item in data]
    if type(data) is not list:
        data = data.tolist()
    if type(data) in [int, torch.int, str, np.str_]:
        data = [data]
    # decode if elements are not already strings, or lists of strings (which would suggest it had been decoded already)
    need_to_decode = not (type(data[0]) is str or type(data[0]) is np.str_ or (type(data) is list and type(data[0][0]) is str))
    if need_to_decode:
        data = [tokenizer.decode(safe_seq(seq), skip_special_tokens=True) for seq in data]
    # lowercase and strip the strs
    multiple_eligible_labels = type(data[0]) is list
    if multiple_eligible_labels:
        data = [[x.lower().strip().replace('.', '') for x in eligible_labels] for eligible_labels in data]
    else:
        data = [x.lower().strip().replace('.', '') for x in data]
    # convert to np array or list of lists
    if type(data) is torch.Tensor:
        data = data.detach().cpu().numpy()
    elif type(data) is list and type(data[0]) is list:
        data = data # skip the array formatting here as it will not be used in downstream metrics
    else:
        data = np.array(data)
    return data

def compute_acc_sum(tokenizer, preds, labels, return_where_correct=False):
    """
    Computes # correct in a batch of preds, for str preds and labels.
    Pred is correct if the label appears within the pred str.
    Optionally returns binary vector of prediction accuracies.

    args:
        preds and labels should be list, 1-d np.array, or 1-d torch.tensor of ints or strs
    """
    preds = force_not_dimensionless(preds) # dimensionless happens when using one_d_tensor[int] slicing
    labels = force_not_dimensionless(labels)
    preds = standardize_preds_or_labels(preds, tokenizer)
    labels = standardize_preds_or_labels(labels, tokenizer)
    assert len(preds) > 0
    assert len(preds) == len(labels), "len of preds and labels not equal"
    many_eligible_labels = type(labels[0]) is list
    if not many_eligible_labels:
        # binary_correct = preds==labels
        binary_correct = np.array([label in pred for pred, label in zip(preds, labels)])
    else:
        # binary_correct = np.array([pred in eligible_labels for pred, eligible_labels in zip(preds, labels)])
        binary_correct = np.array([any([label in pred for label in eligible_labels]) for pred, eligible_labels in zip(preds, labels)])
    acc_sum = np.sum(binary_correct)
    if not return_where_correct:
        return acc_sum
    else:
        return (acc_sum, binary_correct)