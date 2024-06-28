'''
filter wikidata5m to a subset of the graph
'''
import os
import numpy as np
import random
import json
import jsonlines
import sys
import time
from collections import OrderedDict
from copy import deepcopy
import torch

from utils import chunk_array, min_max_mean
import data_utils
import LM_utils
import utils
import data_classes
from data_classes.knowledge_graph import KnowledgeGraph

def read_wikidata5m(data_dir, read_n_facts=None, exclude_entities=None, exclude_relations=None):
    """
    returns:
        all_data: list of individual facts (individual fact = dict)
        entity_dict: entity code to entity strs
        relation_dict: rel code to rel strs
    """
    # make entity and relation dictionaries
    print("Reading entities and relations...")
    entity_dict = OrderedDict()
    relation_dict = OrderedDict()
    entity_path = os.path.join(data_dir, 'wikidata5m_entity.txt')
    relation_path = os.path.join(data_dir, 'wikidata5m_relation.txt')
    with open(entity_path, 'r') as file:
        for line in file:
            identifier = line.split()[0]
            entities = [text.strip('\n') for text in line.split('\t')[1:]]
            entity_dict[identifier] = entities
    with open(relation_path, 'r') as file:
        for line in file:
            identifier = line.split()[0]
            relations = [text.strip('\n') for text in line.split('\t')[1:]]
            relation_dict[identifier] = relations
    # read all data
    print("Reading all facts...")
    load_path = os.path.join(data_dir, f"wikidata5m_transductive_train.txt")
    all_data = []
    with open(load_path, 'r') as file:
        for num, line in enumerate(file):
            e1, rel, e2 = [text.strip('\n') for text in line.split('\t')]        
            # skip missing ents/rels. there are many missing entities / relations. they are missing from the entity and relation files
            print_missing = False
            if e1 not in entity_dict: 
                if print_missing:
                    print(f"missing {e1} from entity dictionary")
                continue
            if e2 not in entity_dict:
                if print_missing:
                    print(f"missing {e2} from entity dictionary")
                continue
            if rel not in relation_dict: 
                if print_missing:
                    print(f"missing {rel} from relation dictionary")
                continue
            # skip exclude entities/relatinos
            if exclude_entities is not None:
                if e1 in exclude_entities or e2 in exclude_entities:
                    continue
            if exclude_relations is not None:
                if rel in exclude_relations:
                    continue
            # accumulate fact
            new_data = {
                'entity1' : e1,
                'entity2': e2,
                'relation': rel,
            }
            all_data.append(new_data)
            # print progress
            if num % 1000000 == 0 and num != 0:
                print(f" Read {num} points")
            if read_n_facts is not None and num >= read_n_facts - 1:
                break
    print(f"Collected {len(all_data)} points!")
    return all_data, entity_dict, relation_dict

def make_entity_info_dict(facts_list):
    # make entity dictionary. includes r,e2 pairs for each e1
    print("Making entity info dict...")
    entity_info_dict = OrderedDict()
    ent_counter = 0
    for datapoint in facts_list:
        e1 = datapoint['entity1']
        e2 = datapoint['entity2']
        rel = datapoint['relation']
        if e1 not in entity_info_dict:
            entity_info_dict[e1] = {
                'id': ent_counter,
                'rel_to_e2s': OrderedDict(),
                'rels_list': [],
            }
            entity_info_dict[e1]['rel_to_e2s'][rel] = [e2]
            entity_info_dict[e1]['rels_list'].append(rel)
            ent_counter += 1
        else:
            if rel not in entity_info_dict[e1]['rel_to_e2s']:
                entity_info_dict[e1]['rel_to_e2s'][rel] = [e2]
                entity_info_dict[e1]['rels_list'].append(rel)
            # COMBINE ANSWERS FROM 1:MANY RELATIONS INTO A SINGLE RELATION'S LIST
            else:
                entity_info_dict[e1]['rel_to_e2s'][rel].append(e2)
    return entity_info_dict

def find_relation_coccurence(entity_info_dict, relations_list, min_relations_per_entity=1):
    # get matrix of co-occurence statistics for relations
    # i.e., for a given entity, if it has one relation, % chance it has the others?
    print("Computing relation co-occurences...")
    num_rels = len(relations_list)
    C = np.zeros((num_rels, num_rels))
    for e1, info in entity_info_dict.items():
        if 'rels_list' in info:
            unique_rels = info['rels_list']
        else:
            unique_rels = list(info.keys())
        # don't count the relation co-occurence if the entity doesn't have enough of them to later be collected
        if len(unique_rels) >= min_relations_per_entity:
            for rel1 in unique_rels:
                for rel2 in unique_rels:
                    rel1_idx = relations_list.index(rel1)
                    rel2_idx = relations_list.index(rel2)
                    C[rel1_idx, rel2_idx] += 1
    C_normalized = C.copy()
    for i in range(num_rels):
        for j in range(num_rels):
            if i == j:
                C_normalized[i,j] = 0 # zero out diagonal, raw counts will be in C
    # do division by rows because some rows could have zeros
    for i in range(num_rels):        
        row_sum = C_normalized[i].sum()
        C_normalized[i] = C_normalized[i] / row_sum if row_sum > 0 else np.zeros(num_rels)
    return C_normalized, C

def compute_num_facts(entity_info_dict):
    # compute number of available 1:1 (e,r,o) facts available
    num_facts = 0
    for e1, info_dict in entity_info_dict.items():
        num_facts += len(info_dict)
    return num_facts

def compute_summary_stats(knowledge_graph, relations, entity_dict, relation_dict, save_dir=None, 
                          tokenizer=None, C_normalized=None, C=None, compute_conditionals=False):
    """
    print stats:
    - Num subject entities
    - Num relations
    - Num object entities
    - Num facts
    - Avg # rels per entitiy
    - Distribution of # objects per rel
    - Frequency of subject entities in atomic facts
    - Frequency of object entities in atomic facts
    - Frequency of relations in atomic facts
    - co-occurence statistics matrix, and raw counts
    """
    subject_entities = list(knowledge_graph.all_subject_entities)
    entity_info_dict = knowledge_graph.entity_info_dict
    object_entities = set()
    rels_per_entity = []
    properties = set()
    num_facts = 0
    num_words = 0
    num_tokens = 0
    for e1, info_dict in entity_info_dict.items():
        rels_per_entity.append(len(info_dict))
        for rel, e2 in info_dict.items():
            object_entities.add(e2)
            num_facts += 1
            properties.add(f"{rel} {e2}")
            fact = {'e1': e1, 'rel': rel, 'e2': e2}
            verbalized_fact = data_utils.verbalize_fact(fact, entity_dict, relation_dict)
            num_words += len(verbalized_fact.split())
            if tokenizer is not None:
                num_tokens += len(tokenizer.encode(verbalized_fact, add_special_tokens=False))
    object_entities = sorted(list(object_entities))
    e1_appearances = {e1: 0 for e1 in subject_entities}
    rel_appearances = {rel: 0 for rel in relations}
    e2_appearances = {e2: 0 for e2 in object_entities}
    for fact in knowledge_graph.unroll_atomic_facts():
        e1 = fact['e1']
        rel = fact['rel']
        e2 = fact['e2']
        e1_appearances[e1] += 1
        rel_appearances[rel] += 1
        e2_appearances[e2] += 1
    e1_counts = [(entity_dict[e1][0], count, e1) for e1, count in e1_appearances.items()]
    rel_counts = [(relation_dict[rel][0], count, rel) for rel, count in rel_appearances.items()]
    e2_counts = [(entity_dict[e2][0], count, e2) for e2, count in e2_appearances.items()]
    if C_normalized is None or C is None:
        C_normalized, C = find_relation_coccurence(entity_info_dict, relations_list = relations)
    # now print everything
    num_relations = len(relations)
    num_object_entities = len(object_entities)
    print("Num facts: ", num_facts)
    print("Num subject entities: ", len(subject_entities))
    print("Num relations: ", num_relations)
    print("Num object entities: ", num_object_entities)
    print("Num properties: ", len(properties))
    print("Num words: ", utils.format_large_number(num_words))
    print("Num tokens: ", utils.format_large_number(num_tokens))
    print("Num overlap between subj and obj entities: ", len(np.intersect1d(subject_entities, list(object_entities))))
    print(f"Num rels per entity: {min_max_mean(rels_per_entity)}")
    print_examples = 3
    for name, counts_list in zip(
        ['subject entities', 'relations', 'object entities'],
        [e1_counts, rel_counts, e2_counts]
    ):
        sorted_counts = sorted(counts_list, key=lambda x: -x[1])
        print(f"Top/bottom {name} frequencies:")
        for x_str, count, x in sorted_counts[:print_examples]:
            print(f" {x_str[:20]:20s} ({x}) - prop: {count/num_facts*100:.2f}%, count: {count}")
        print(' ...')
        for x_str, count, x in sorted_counts[-print_examples:]:
            print(f" {x_str[:20]:20s} ({x}) - prop: {count/num_facts*100:.2f}%, count: {count}")
    # print("Relation co-occurence statistics")
    for rel_num, rel in enumerate(relations):
        freqs = [f"{col_num}: {100*x:2.0f}%" for col_num, x in enumerate(C_normalized[rel_num])]
        print(f" ({rel_num}) {relation_dict[rel][0][:20]:20s}: {' | '.join(freqs)}")
    # print("Relation co-occurence counts")
    for rel_num, rel in enumerate(relations):
        counts = [f"{col_num}: {int(x):4d}" for col_num, x in enumerate(C[rel_num])]
        print(f" ({rel_num}) {relation_dict[rel][0][:20]:20s}: {' | '.join(counts)}")
    if save_dir is not None:
        save_path = os.path.join(save_dir, f"co-occurence_statistics_n{num_facts}.csv")
        np.savetxt(save_path, 100*C_normalized, delimiter=',')
        save_path = os.path.join(save_dir, f"co-occurence_counts_n{num_facts}.csv")
        np.savetxt(save_path, C, delimiter=',')
    '''
    compute some conditional probabilities
    - if an entity has rel-A to e2-A, then it will have a conditional distribution over other relations and objects
    - want to compute probability that it has rel-B and e2-B, vs. probability that is has e2-B conditioned on it having rel-B
    '''
    if compute_conditionals:
        rel_obj_to_counts_matrix_dict = OrderedDict()
        rels_and_objects = set()
        property_to_entities_with_property = {}
        for e1, info_dict in entity_info_dict.items():
            for rel, e2 in info_dict.items():
                rel_and_object_str = f"{relation_dict[rel][0]} = {entity_dict[e2][0]}"
                rels_and_objects.add(rel_and_object_str)
                # add e1 to list of entities with that property 
                if rel_and_object_str not in property_to_entities_with_property:
                    property_to_entities_with_property[rel_and_object_str] = set([e1])
                else:
                    property_to_entities_with_property[rel_and_object_str].add(e1)
        print(f"Finding conditional distribution stats for {len(rels_and_objects)} (., relation, object) properties...")
        for rel_and_object in rels_and_objects:
            rel_by_object_counts = np.zeros((num_relations, num_object_entities))
            rel_obj_to_counts_matrix_dict[rel_and_object] = rel_by_object_counts
        for e1, info_dict in entity_info_dict.items():
            all_rels_and_objects = [(rel, e2) for rel, e2 in info_dict.items()]
            for rel_and_object_1 in all_rels_and_objects:
                for rel_and_object_2 in all_rels_and_objects:
                    if rel_and_object_1 != rel_and_object_2:
                        rel, e2 = rel_and_object_1
                        rel_and_object_str = f"{relation_dict[rel][0]} = {entity_dict[e2][0]}"
                        rel, e2 = rel_and_object_2
                        rel_idx, e2_idx = relations.index(rel), object_entities.index(e2)
                        rel_obj_to_counts_matrix_dict[rel_and_object_str][rel_idx][e2_idx] += 1
        # normalize counts in distr dict
        rel_obj_to_statistics = OrderedDict()
        for rel_and_object, distr in rel_obj_to_counts_matrix_dict.items():
            # each row represents a relation, columns are counts of how often each object appears. so, normalize rows
            raw_counts = distr.copy()
            for i in range(distr.shape[0]):
                if distr[i].sum() >= 1:
                    distr[i] = distr[i] / distr[i].sum()
            rel_obj_to_statistics[rel_and_object] = {
                'counts': raw_counts,
                'freqs': distr,
            }
        # collect example conditional distributions
        rel_strs = [relation_dict[rel][0] for rel in relations] # need this below
        conditional_examples = []
        # min examples of "property1 implies property2 with probability p"
        # NOTE property2 means (s,r,o), not just has that relation. prob has relation is given by rel_2_probability)
        min_examples_of_conditional_relationship = 5 
        for rel_and_object_1, stats in rel_obj_to_statistics.items():
            counts_mat = stats['counts']
            freqs_mat = stats['freqs']
            for rel_idx in range(freqs_mat.shape[0]):
                rel_2 = relation_dict[relations[rel_idx]][0]
                rel_1_idx = rel_strs.index(rel_and_object_1.split('=')[0].strip()) # forgot to carry forward, need to get the first property's relation index
                rel_2_probability = C_normalized[rel_1_idx, rel_idx]
                counts = counts_mat[rel_idx]
                distr = freqs_mat[rel_idx]
                e2_idx = np.argmax(distr)
                max_freq = distr[e2_idx]
                num_examples = int(counts[e2_idx])
                example = (rel_and_object_1, rel_2, rel_2_probability, counts, distr, max_freq, num_examples)
                if num_examples >= min_examples_of_conditional_relationship:
                    conditional_examples.append(example)
        sorted_counts = sorted(conditional_examples, key=lambda x: -x[-2]) # sort by max_freq, descending
        if len(sorted_counts) > 0:
            print(f"There are {len(sorted_counts)} total conditional relationships with at least {min_examples_of_conditional_relationship} examples of the modal property2")
            print(f"Top conditional frequencies:")
            distr_cutoff = 5
            print_examples = 3
            for example_num, (property1, rel_2, rel_2_probability, counts, distr, max_freq, count) in enumerate(sorted_counts[:print_examples]):
                property1_probability = len(property_to_entities_with_property[property1]) / len(subject_entities)
                print(f" example {example_num}: '{property1}' ({100*property1_probability:.2f}% of entities have this property)")
                print(f"   has '{rel_2}' with probability {100*rel_2_probability:.2f}%. IF HAS that relation, implies one of the following object with probability...")
                sorted_idx = np.argsort(-distr)
                sorted_distr = distr[sorted_idx]
                formatted_distr = [f"{x*100:.1f}%" for x in sorted_distr]
                ent_strs = [entity_dict[object_entities[idx]][0] for idx in sorted_idx]
                zipped_list = list([f"{name}: {prob}" for prob, name in zip(formatted_distr, ent_strs)])
                print(f"   distr: {zipped_list[:distr_cutoff]}...")
                print(f"   with {count} examples of the modal conditional property")
    print("\nDownstream training info...")
    print("Num facts: ", utils.format_large_number(num_facts))
    print("Num words: ", utils.format_large_number(num_words))
    print("Num tokens: ", utils.format_large_number(num_tokens))
    return

def filter_alias_dict(alias_dict, model, tokenizer, batch_size):
    # there are many aliases for each entity and relation. we can choose one alias per ent/rel by selecting the highest probability entry
    if not tokenizer.pad_token_id:
        if tokenizer.eos_token:
            tokenizer.add_special_tokens({'pad_token' : tokenizer.eos_token})
        else:
            tokenizer.add_special_tokens({'pad_token' : tokenizer.unk_token})
    num_items = len(alias_dict)
    
    for i, (code, aliases) in enumerate(alias_dict.items()):
        alias_probs = {}
        # filter to aliases with >= 3 characters
        if any([len(alias) >= 3 for alias in aliases]):
            aliases = [alias for alias in aliases if len(alias) >= 3]
        # test no more than 20 for each entry
        if len(aliases) > 20:
            aliases = aliases[:20]
        # chunking into batches
        for chunk in chunk_array(aliases, batch_size):
            sentences = chunk
            batch = tokenizer(sentences, return_tensors='pt', add_special_tokens=False, padding=True)
            utils.move_kwargs_to_gpu(batch)
            with torch.no_grad():
                probs = LM_utils.compute_probs_from_batch(model, batch, return_value='probs', pad_token_id=tokenizer.pad_token_id)
            for alias, prob in zip(chunk, probs):
                alias_probs[alias] = prob
            del probs
        max_prob = max(list(alias_probs.values()))
        max_prob_candidates = [k for k in alias_probs.keys() if alias_probs[k] == max_prob]
        assert len(max_prob_candidates) > 0, f"issue with probs computation: {list(alias_probs.values())}"
        # break ties by number of capital letters (many entities/relations are proper nouns)
        def count_capital_letters(string):
            return sum([letter == letter.upper() for letter in string])
        if len(max_prob_candidates) > 1:
            sorted_by_str_len = sorted([(alias, count_capital_letters(alias)) for alias in max_prob_candidates], key=lambda x: x[1])
        alias_dict[code] = [max_prob_candidates[0]]
        # print example
        if i < 5:
            print("Filtering ent/rel aliases as follows: ")
            for alias, prob in sorted(alias_probs.items(), key= lambda x: -x[1]):
                print(f" alias: {alias[:25]:25s} has prob: {prob}")
        if num_items > 500 and i % 500 == 0:
            print(f"progress: {i}/{num_items}")
    return alias_dict

def make_wikidata_KG(args,
                    data_dir,
                    min_relations_per_entity=2,
                    min_relation_cooccurences=50000,
                    top_k_relations=30,
                    model=None,
                    tokenizer=None,
                    batch_size=None,
                    read_n_facts=None,
                    overwrite_cached_data=False,
                    verbose=False,
    ):
    """
    filters wikidata5m to a subgraph based on args, save resulting KG class as pkl with np.save
    args:
        data_dir: path to folder with Wikidata5m train data
        min_relations_per_entity: to keep a subject entity, min number of relations it must have        
        min_relation_cooccurences: require selected relations to co-occur at least this many times (out of about 17m facts after initial entity/rel filtering)
        top_k_relations: only use the top k relations in the wikidata graph
        model: language model used for picking verbalizations of entity/relation codes. use more "typical" (i.e. high prob) aliases
    """    
    start = time.time()

    EXCLUDE_RELATIONS = [
        'P735', # given name. too easy to read off of subject entity
        'P131', # located in the administrative territorial entity
    ]
    EXCLUDE_ENTITIES = [
        'Q16521', # 'taxon', usually redundant with genus, comprises up to 30% of all object entities if included
        'Q5', # 'human', ends up being ~23% of all object entities if included
    ]

    # load data
    all_data, entity_dict, relation_dict = read_wikidata5m(data_dir, 
                                                           read_n_facts=read_n_facts,
                                                           exclude_entities=EXCLUDE_ENTITIES,
                                                           exclude_relations=EXCLUDE_RELATIONS)
    print("Excluded the following entities: ")
    for e1 in EXCLUDE_ENTITIES:
        print(entity_dict[e1][:5])
    print("Excluded the following relations: ")
    for rel in EXCLUDE_RELATIONS:
        print(relation_dict[rel][:5])

    # make entity info dict (knowledge graph) out of list of individual facts
    entity_info_dict_path = os.path.join(data_dir, f"entity_info_dict_read{read_n_facts}.npy")
    if not os.path.exists(entity_info_dict_path):
        entity_info_dict = make_entity_info_dict(all_data)
        np.save(entity_info_dict_path, entity_info_dict)
    else:
        entity_info_dict = np.load(entity_info_dict_path, allow_pickle=True).item()

    # make co-occurence for relations
    C_path = os.path.join(data_dir, f"co-occurence_read{read_n_facts}_min-rel{min_relations_per_entity}.npy")
    if not os.path.exists(C_path) or overwrite_cached_data:
        relations_list = list(relation_dict.keys())
        C_normalized, C = find_relation_coccurence(entity_info_dict, relations_list, min_relations_per_entity=min_relations_per_entity)
        np.save(C_path, {'C_normalized': C_normalized, 'C': C})
    else:
        C_dict = np.load(C_path, allow_pickle=True).item()
        C_normalized, C = C_dict['C_normalized'], C_dict['C']

    # filter relations. relation pairs must have at least min_cooccured_relations cocurrences
    # convert C_normalized and C to list of stats with indices
    min_cooccurences = min_relation_cooccurences
    cooccurence_stats = []
    relations_list = list(relation_dict.keys())
    top_k = top_k_relations
    for i in range(C_normalized.shape[0]):
        for j in range(C_normalized.shape[1]):
            if i != j:
                co_occ = C_normalized[i,j]
                count = C[i,j]
                stats = ((i,j), co_occ, count)
                cooccurence_stats.append(stats)
    cooccurence_stats = sorted(cooccurence_stats, key=lambda x : -x[2]) # sort by most-to-least raw count
    cooccurence_stats = [cooccurence_stats[i] for i in range(0, len(relations_list), 2)] # take every other one due to (r1,r2)=(r2,r1) counting symmetry
    filtered_cooccurence_stats = [x for x in cooccurence_stats if x[2] >= min_cooccurences]
    # check that we have enough relations
    collected_relations = set()
    for indices, _, _ in filtered_cooccurence_stats:
        collected_relations.add(indices[0])
        collected_relations.add(indices[1])
    assert len(collected_relations) >= top_k, f"Only have {len(collected_relations)} relations with {min_cooccurences}. Decrease min_cooccurences to have at least {top_k} relations (or read more data)"
    # print examples
    print("Most frequently co-occuring relations: ")
    print_examples = 3
    for indices, co_occ, count in cooccurence_stats[:print_examples]:
        r1, r2 = relations_list[indices[0]], relations_list[indices[1]]
        r1_str, r2_str = relation_dict[r1][0], relation_dict[r2][0]
        print(f" {r1_str[:20]:20s} | {r2_str[:20]:20s} - cond prob: {co_occ*100:.2f}, raw count: {int(count)}")
    print("Least frequently co-occuring relations: ")
    for indices, co_occ, count in cooccurence_stats[-print_examples:]:
        r1, r2 = relations_list[indices[0]], relations_list[indices[1]]
        r1_str, r2_str = relation_dict[r1][0], relation_dict[r2][0]
        print(f" {r1_str[:20]:20s} | {r2_str[:20]:20s} - cond prob: {co_occ*100:.2f}, raw count: {int(count)}")
    print(f"We have {len(collected_relations)} relations with at least {min_cooccurences} co-occured samples")
    
    # get top k unique relations by co-occurence counts
    print(f"Filtering to relations in top {top_k} relations with at least {min_cooccurences} co-occurences...")
    top_relations = []
    while len(top_relations) < top_k and len(cooccurence_stats) > 0:
        indices, co_occ, count = cooccurence_stats.pop(0)
        r1, r2 = relations_list[indices[0]], relations_list[indices[1]]
        if r1 not in top_relations and r2 not in top_relations:
            want_to_add = 2
        elif r1 not in top_relations or r2 not in top_relations:
            want_to_add = 1
        else:
            continue
        have_room_for_two = len(top_relations) < top_k - 1
        if want_to_add == 2 and have_room_for_two:
            top_relations.append(r1)
            top_relations.append(r2)
        if want_to_add == 2 and not have_room_for_two:
            continue
        if want_to_add == 1:
            if r1 not in top_relations:
                top_relations.append(r1)
            else:
                top_relations.append(r2)

    # filter entity_info_dict...
    # use only top-k relations
    # require min_relations_per_entity
    # if a fact is 1:N, reduce it to 1:1 
    # THIS STEP INVOLVES CHANGING THE NAMING SCHEME. no more rel_to_e2s and rel_list. we will only make rel_to_e2
    filtered_entity_info_dict = dict()
    current_relations = []
    for e1, info in entity_info_dict.items():
        unique_rels = info['rels_list']
        # check if have enough relations for min_relations_per_entity
        num_rels_in_requested_rels = sum([rel in top_relations for rel in unique_rels])
        if num_rels_in_requested_rels >= min_relations_per_entity:
            pull_rels = [rel for rel in unique_rels if rel in top_relations]
            rel_to_e2 = {}
            for rel in pull_rels:
                e2s = info['rel_to_e2s'][rel]
                choose_e2 = e2s[0] # arbitrarily take first object
                rel_to_e2[rel] = choose_e2
            # add fact to filtered entity info dict
            filtered_entity_info_dict[e1] = rel_to_e2
            # get current relations
            for rel in pull_rels:
                if rel not in current_relations:
                    current_relations.append(rel)
    # check that we have the right number of relations after filtering
    assert len(set(current_relations)) == top_k
    print(" Used relations: ", sorted(current_relations))
    print(" Used relations: ", sorted([relation_dict[rel][0] for rel in current_relations]))

    # check how many facts we have based on this knowledge graph
    available_facts = compute_num_facts(filtered_entity_info_dict)
    print(f" Num. available facts: {available_facts}")

    # construct knowledge graph (basically equal to filtered_entity_info_dict)
    knowledge_graph = KnowledgeGraph(entity_info_dict=filtered_entity_info_dict)
    
    # # first add a subset of entities with complete coverage of the relations with at least min_examples_per_relation. ensure that each relation is used in at least .1% of facts
    # min_examples_per_relation = int(np.ceil(.001 * available_facts))
    # entities_for_relation_coverage = []
    # # make rel to e1s dict
    # relation_info_dict = OrderedDict()
    # for e1, info_dict in filtered_entity_info_dict.items():
    #     # will add this entity if it has a relation we need to cover
    #     for rel, e2 in info_dict.items():
    #         if rel not in relation_info_dict:
    #             relation_info_dict[rel] = [e1]
    #         else:
    #             relation_info_dict[rel].append(e1)
    # # get e1s for relation coverage
    # for rel, e1s in relation_info_dict.items():
    #     assert len(e1s) >= min_examples_per_relation, f"Want {min_examples_per_relation} examples for rel {rel}, but only have {len(e1s)} known subjects with this relation"
    #     add_e1 = np.random.choice(e1s, size=min_examples_per_relation).tolist()
    #     entities_for_relation_coverage.extend(add_e1)
    # entities_for_relation_coverage = sorted(list(set(entities_for_relation_coverage)))
    # print(f"Starting with {len(entities_for_relation_coverage)} seed entities for relation coverage")
    # # add all facts about these entities to final_facts (removing as we go)
    # for e1 in entities_for_relation_coverage:
    #     info_dict = filtered_entity_info_dict.pop(e1)
    #     facts = [{'e1': e1, 'rel': rel, 'e2': e2} for (rel, e2) in info_dict.items()]
    #     for fact in facts:
    #         knowledge_graph.add_fact(fact)
    #         if knowledge_graph.num_facts == num_facts:
    #             break
    #     if knowledge_graph.num_facts == num_facts:
    #         break

    # # second, start adding entities until we hit the requested number of facts. To increase graph density, sort by number of relations per entity
    # e1_relation_count_list = [(e1, len(info_dict)) for e1, info_dict in filtered_entity_info_dict.items()]
    # e1_relation_count_list = sorted(e1_relation_count_list, key= lambda x: -x[1])
    # for (e1, count) in e1_relation_count_list:
    #     info_dict = filtered_entity_info_dict.pop(e1)
    #     facts = [{'e1': e1, 'rel': rel, 'e2': e2} for (rel, e2) in info_dict.items()]
    #     for fact in facts:
    #         knowledge_graph.add_fact(fact)
    #         if knowledge_graph.num_facts == num_facts:
    #             break
    #     if knowledge_graph.num_facts == num_facts:
    #         break

    # check knowledge_graph fact count
    assert knowledge_graph.compute_num_facts() == knowledge_graph.num_facts, "KG num_facts_counter disagrees with final count of added facts"

    # filter entity and relation verbalization dicts to use highest probability aliases under a language model
    seen_entitities = knowledge_graph.all_entities
    seen_relations = knowledge_graph.all_relations
    entity_dict = {k:v for k,v in entity_dict.items() if k in seen_entitities}
    relation_dict = {k:v for k,v in relation_dict.items() if k in seen_relations}
    filtered_entity_dict_path = os.path.join(data_dir, f"filtered_ent_dict_read-{read_n_facts}.npy")
    filtered_relation_dict_path = os.path.join(data_dir, f"filtered_rel_dict_read-{read_n_facts}.npy")
    if model is not None and tokenizer is not None:
        if not os.path.exists(filtered_entity_dict_path) or overwrite_cached_data:
            print("Selecting aliases for entities based on an LM prob...")
            entity_dict = filter_alias_dict(entity_dict, model, tokenizer, batch_size)
            np.save(filtered_entity_dict_path, entity_dict)
        if not os.path.exists(filtered_relation_dict_path) or overwrite_cached_data:
            print("Selecting aliases for relations based on an LM prob...")
            relation_dict = filter_alias_dict(relation_dict, model, tokenizer, batch_size)
            np.save(filtered_relation_dict_path, relation_dict)
    elif os.path.exists(filtered_entity_dict_path):
        print(f"Loading entity dict from {filtered_entity_dict_path}...")
        entity_dict = np.load(filtered_entity_dict_path, allow_pickle=True).item()
        print(f"Loading relation dict from {filtered_entity_dict_path}...")
        relation_dict = np.load(filtered_relation_dict_path, allow_pickle=True).item()
    else:
        pass

    if verbose:
        print("Example facts: ")
        for e1 in knowledge_graph.all_subject_entities:
            facts = [{'e1': e1, 'rel': rel, 'e2': e2} for (rel, e2) in knowledge_graph.entity_info_dict[e1].items()]
            rels = [rel for (rel, e2) in knowledge_graph.entity_info_dict[e1].items()]
            if 'parent taxon' in rels:
                print(knowledge_graph.entity_info_dict[e1])
                breakpoint()
        
        example_counter, example_facts = 0, 10
        for e1 in knowledge_graph.all_subject_entities:
            facts = [{'e1': e1, 'rel': rel, 'e2': e2} for (rel, e2) in knowledge_graph.entity_info_dict[e1].items()]
            for fact in facts:
                e1_str = entity_dict[fact['e1']][0]
                rel_str = relation_dict[fact['rel']][0]
                e2_str = entity_dict[fact['e2']][0]
                print(f" {e1_str[:20]:20s} | {rel_str[:20]:20s} | {e2_str[:20]:20s}")
                example_counter += 1
                if example_counter == example_facts:
                    break
            if example_counter == example_facts:
                break

    # COMPUTE SUMMARY STATISTICS (and print)
    C_path = os.path.join(data_dir, f"filtered_co-occurence_read-{read_n_facts}_min-rel-{min_relations_per_entity}.npy")
    relations_list = knowledge_graph.relations_list
    if not os.path.exists(C_path) or overwrite_cached_data:
        C_normalized, C = find_relation_coccurence(knowledge_graph.entity_info_dict, relations_list, min_relations_per_entity=min_relations_per_entity)
        np.save(C_path, {'C_normalized': C_normalized, 'C': C})
    else:
        C_dict = np.load(C_path, allow_pickle=True).item()
        C_normalized, C = C_dict['C_normalized'], C_dict['C']
    if verbose:
        compute_summary_stats(knowledge_graph, relations_list, entity_dict, relation_dict, save_dir=data_dir, tokenizer=tokenizer, C_normalized=C_normalized, C=C)

    # translate graph to english
    knowledge_graph.set_alias_dicts(entity_dict, relation_dict)
    knowledge_graph.translate_into_english()
    # cap object entity num tokens
    if tokenizer is not None and args.max_object_tokens >= 1:
        print(f"Cutting object entities from their token length to a max token length of {args.max_object_tokens}...")
        knowledge_graph.cap_object_entity_token_len(tokenizer, args.max_object_tokens)

    # save knowledge graph
    save_path = os.path.join(data_dir, f"knowledge_graph_read-{read_n_facts}.npy")
    print(f"Saving knowledge graph at {save_path}")
    np.save(save_path, knowledge_graph)

    print(f"\n Overall runtime: {(time.time() - start) / 3600:.2f} hours")
    return knowledge_graph