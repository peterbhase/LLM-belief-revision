import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def verbalize_fact(fact, entity_dict, relation_dict):
    e1_str = entity_dict[fact['e1']][0]
    rel_str = relation_dict[fact['rel']][0]
    e2_str = entity_dict[fact['e2']][0]
    fact_str = f"{e1_str} {rel_str} {e2_str}"
    return fact_str

def state_P_is_true_false(sentence, truth_value, probability=None):
    assert truth_value in ['true', 'false']
    if probability is not None:
        assert probability % .1 == 0, "please use multiples of 0.1 for probabilities"
    statement = f"\"{sentence}\" is {truth_value}"
    if probability is not None:
        statement += f" with probability {probability}"
    return statement

def connect_sentences(sentence1, sentence2, connective):
    '''
    connect two sentences with a specified connective. sentences do not need to be atomic
    '''
    assert connective in ['or', 'and', 'if-then', 'not']
    if connective == 'not':
        assert sentence2 is None, "sentence2 provided to connect_sentences but the unary connective is 'not'"
        new_sentence = f"not {sentence1}"
    else:
        assert sentence2 is not None, "need to supply sentence2 if a binary connective used"
        if connective == 'if-then':
            new_sentence = f"if {sentence1} then {sentence2}"
        else:
            new_sentence = f"{sentence1} {connective} {sentence2}"
    return new_sentence

def get_random_fact(entity_info_dict):
    # Returns a fact that is directly contradictory to a known training fact. 
    # Uses a random entity1 if 'fact' not provided. 
    all_entities = list(entity_info_dict.keys())
    entity1 = np.random.choice(all_entities)
    rels_list = entity_info_dict[entity1]['rels_list']
    relation = np.random.choice(rels_list)
    fact = {
        'e1': entity1,
        'rel': relation,
        'e2': entity_info_dict[entity1]['rel_to_e2s'][relation][0]
    }
    return fact

def get_all_object_entities(entity_info_dict):
    object_entities = set()
    for e1, info in entity_info_dict.items():
        for rel, e2s in info['rel_to_e2s'].items():
            for e2 in e2s:
                object_entities.add(e2)
    return list(object_entities)

def get_false_fact(entity_info_dict, object_entities=None, fact=None):
    # Returns a fact that is directly contradictory to a known training fact. 
    # Uses a random entity1 if 'fact' not provided. 
    all_entities = list(entity_info_dict.keys())
    if object_entities is None:
        object_entities = get_all_object_entities(entity_info_dict)
    if fact is None:
        entity1 = np.random.choice(all_entities)
        rels_list = entity_info_dict[entity1]['rels_list']
        relation = np.random.choice(rels_list)
        entity2 = entity_info_dict[entity1]['rel_to_e2s'][relation][0]
    else:
        entity1 = fact['e1']
        relation = fact['rel']
        entity2 = fact['e2']
    eligible_entities = np.setdiff1d(object_entities, [entity2])
    new_entity2 = np.random.choice(eligible_entities)
    new_fact = {
        'e1': entity1,
        'rel': relation,
        'e2': new_entity2
    }
    return new_fact

def get_one_TF_sentence(fact, entity_info_dict, entity_dict, relation_dict, object_entities):
    # returns constructed 'complex' sentence
    get_true = (np.random.random() > .5)
    if get_true:
        fact = verbalize_fact(fact, entity_dict, relation_dict)
        new_sentence = state_P_is_true_false(fact, truth_value='true')
    else:
        false_fact = get_false_fact(entity_info_dict, object_entities, fact)
        new_sentence = verbalize_fact(false_fact, entity_dict, relation_dict)
        new_sentence = state_P_is_true_false(new_sentence, truth_value='false')
    return new_sentence

def get_one_or_sentence(fact, entity_info_dict, entity_dict, relation_dict, object_entities):
    # returns constructed 'complex' sentences
    sent1_true = (np.random.random() > .5)
    sent2_true = (np.random.random() > .5)
    if sent1_true:
        sentence1 = verbalize_fact(fact, entity_dict, relation_dict)
    else:
        sentence1 = get_false_fact(entity_info_dict, object_entities, fact)
        sentence1 = verbalize_fact(sentence1, entity_dict, relation_dict)
    if sent2_true:
        sentence2 = get_random_fact(entity_info_dict) # ok to take sentence1 again
        sentence2 = verbalize_fact(sentence2, entity_dict, relation_dict)
    else:
        false_fact = get_false_fact(entity_info_dict, object_entities, fact=None) # make a random sentence that is known to be false
        sentence2 = verbalize_fact(false_fact, entity_dict, relation_dict)
    # switch sentence order randomly
    if (np.random.random() > .5):
        sentence1, sentence2 = sentence2, sentence1
    label = "true" if sent1_true or sent2_true else "false"
    new_sentence = connect_sentences(sentence1, sentence2, connective='or')
    new_sentence = state_P_is_true_false(new_sentence, label)
    return new_sentence

def get_one_and_sentence(fact, entity_info_dict, entity_dict, relation_dict, object_entities):
    # returns constructed 'complex' sentences
    sent1_true = (np.random.random() > .5)
    sent2_true = (np.random.random() > .5)
    if sent1_true:
        sentence1 = verbalize_fact(fact, entity_dict, relation_dict)
    else:
        sentence1 = get_false_fact(entity_info_dict, object_entities, fact)
        sentence1 = verbalize_fact(sentence1, entity_dict, relation_dict)
    if sent2_true:
        sentence2 = get_random_fact(entity_info_dict) # ok to take sentence1 again
        sentence2 = verbalize_fact(sentence2, entity_dict, relation_dict)
    else:
        false_fact = get_false_fact(entity_info_dict, object_entities) # make a random sentence that is known to be false
        sentence2 = verbalize_fact(false_fact, entity_dict, relation_dict)
    # switch sentence order randomly
    if (np.random.random() > .5):
        sentence1, sentence2 = sentence2, sentence1
    label = "true" if sent1_true and sent2_true else "false"
    new_sentence = connect_sentences(sentence1, sentence2, connective='or')
    new_sentence = state_P_is_true_false(new_sentence, label)
    return new_sentence
    
def get_one_not_sentence(fact, entity_info_dict, entity_dict, relation_dict, object_entities, always_TF=False):
    # returns constructed 'complex' sentences
    sent_true = (np.random.random() > .5)
    if sent_true:
        sentence = verbalize_fact(fact, entity_dict, relation_dict)
        sentence = connect_sentences(sentence1=sentence, sentence2=None, connective='not')
        sentence = state_P_is_true_false(sentence, 'false')
    else:
        sentence = get_false_fact(entity_info_dict, object_entities, fact)
        sentence = verbalize_fact(sentence, entity_dict, relation_dict)
        sentence = connect_sentences(sentence1=sentence, sentence2=None, connective='not')
        if (np.random.random() > .5) or always_TF: # always use T/F label if turning into prompts
            sentence = state_P_is_true_false(sentence, 'true')
    return sentence

def promptify_fact(fact, entity_dict, relation_dict, use_TF_str=None):
    e1_str = entity_dict[fact['e1']][0]
    rel_str = relation_dict[fact['rel']][0]
    e2_str = entity_dict[fact['e2']][0]
    if use_TF_str is None:
        prompt_str = f"{e1_str} {rel_str}"
        label_str = f"{e2_str}"
    else:
        prompt_str = f"'{e1_str} {rel_str} {e2_str}' is"
        label_str = use_TF_str
    return prompt_str, label_str

def promptify_TF_sentence(sentence):
    label = sentence.split()[-1]
    prompt = sentence.replace(" true", "").replace(" false", "")
    assert label in ['true', 'false']
    return prompt, label

def promptify_TF_sentences(sentences):
    prompts = []
    labels = []
    for sentence in sentences:
        prompt, label = promptify_TF_sentence(sentence)
        prompts.append(prompt)
        labels.append(label)
    return prompts, labels