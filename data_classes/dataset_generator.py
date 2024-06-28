import os
from unittest import TestCase
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import LM_utils
import data_utils
import utils
from utils import min_max_mean
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from copy import deepcopy
import json
import jsonlines as jsonl
import time

# 50 words without prior meaning, appropriate for use as novel entities
nonce_words = [
    "Flim", "Jape", "Vope", "Creb", "Wunx", "Zib", "Plen", "Mib", "Frox", "Grinj",
    "Klon", "Yarp", "Sniv", "Blonk", "Quev", "Dripf", "Skop", "Trin", "Cleez", "Brunt",
    "Flink", "Gwarp", "Jurn", "Plof", "Vint", "Skree", "Blib", "Druv", "Kwim", "Plax",
    "Glibe", "Zonk", "Wimb", "Trop", "Jilk", "Squiv", "Qwep", "Fronz", "Kliv", "Veem",
    "Stib", "Gruf", "Plim", "Junt", "Niv", "Froob", "Blinx", "Dwev", "Klonx", "Quimp"
]


class DatasetGenerator:
    '''
    This class creates a static training corpus for training a logical LM and eval set for model editing evaluation.
    - it relies on a generative model and a knowledge graph
    Here's how we get the total number of facts. Several operations for increasing pretraining data complexity lead to facts appearing multiple times, so we will have to count the final distribution of how often each fact appears.
        n_initial = n_facts * n_base_facts_samples + n_facts * n_complex_sentences_per_fact
        n_total = n_initial + n_sentences * k_homogenous_doc_order_resamples + n_initial * k_heterogenous_doc_resamples
    '''
    def __init__(self, args, pretraining_datapath, eval_datapath, generative_model, 
                 max_sentences_per_doc, n_base_fact_samples, n_complex_sentences_per_fact, k_homogenous_doc_order_resamples, k_heterogenous_doc_order_resamples,
                 complex_sentences_from_noisy_facts=False):
        self.args = args
        self.false_facts_use_prespecified_distractors = args.false_facts_use_prespecified_distractors
        self.generative_model = generative_model
        self.entity_info_dict = generative_model.entity_info_dict
        self.subject_entities = generative_model.subject_entities
        self.relations = generative_model.relations
        self.object_entities = generative_model.object_entities
        self.relevant_relations_dict = generative_model.relevant_relations_dict # this maps from the r to the rel in relevant_property in the distribution p(o|., r, relevant_property)
        self.subjects_in_training_data = list()
        self.objects_in_training_data = list()
        self.max_sentences_per_doc = max_sentences_per_doc
        self.n_base_fact_samples = n_base_fact_samples
        self.n_complex_sentences_per_fact = n_complex_sentences_per_fact if not (self.args.add_is_true_false_sentences and self.args.min_ground_truth_prob < 1) else n_complex_sentences_per_fact - 1
        self.k_homogenous_doc_order_resamples = k_homogenous_doc_order_resamples
        self.k_heterogenous_doc_order_resamples = k_heterogenous_doc_order_resamples
        self.complex_sentences_from_noisy_facts = complex_sentences_from_noisy_facts
        self.pretraining_datapath = pretraining_datapath
        self.eval_datapath = eval_datapath
        self.pretraining_stats_path = pretraining_datapath.replace('.jsonl', '_stats.npy')
        self.all_sampled_facts_path = pretraining_datapath.replace('.jsonl', '_SAMPLED_FACTS.npy')
        self.rng = np.random.default_rng(args.seed)
        # self.relevant_relations_dict_path = pretraining_datapath.replace('.jsonl', '_RELEVANT_RELATIONS.npy')
        # np.save(self.relevant_relations_dict_path, self.relevant_relations_dict)

    def object_idx_to_strs(self, object_idx):
        return np.array([self.object_entities[idx] for idx in object_idx])
    
    def write_pretraining_data(self, n_atomic_facts=None):
        '''
        Write training documents as jsonl file. This file will contain dictionaries, each of which is a training document with metadata
        - uses only n_atomic_facts if provided
        - a fact s,r,o is generated from one of two distributions: p(o|s,r) or p(o|., r, rel_prop)
        - if, in our knowledge_graph, the subject has a relation r and its relevant relation according to the causal graph, we use p(o|., r, rel_prop)
        - otherwise, we use p(o|s,r)
        - homogenous docs will be all about one subject (we add k_homogenous_doc_order_resamples of these)
        - heteorgenous docs will be about multiple subjects (we randomly shuffle all the train sentences then chunk them into docs)
        '''
        homogenous_docs = []
        heterogenous_docs = []
        self.all_sampled_facts_with_weights = []
        self.unique_training_sentences = set()
        self.all_training_sentences = list()
        pretraining_data_stats = {
            'num_true_atomic_facts': 0, 
            'num_atomic_sentences': 0, 
            'num_total_sentences': 0,
            'num_unique_sentences': 0, 
            'num_TF_atomic_sentences': 0,
            'num_not_sentences': 0,
            'num_or_sentences': 0,
            'num_and_sentences': 0,
            'num_if_then_sentences': 0,
        }
        break_counter = 0
        for subj, info in self.generative_model.entity_info_dict.items():
            # first make doc where every sentence includes a sentence about the subject (homogenous doc)
            doc_sentences = []
            base_facts = [] # all true bast facts about the entity
            sampled_facts = [] # list of noisily sampled facts, will be added to doc_facts and used to create complex sentences
            self.subjects_in_training_data.append(subj)
            # sample facts, maybe noisily depending on generative model. This can be not noisy too depending on earlier params
            for rel, _ in info.items():
                true_obj = self.generative_model.get_modal_output(subj, rel)
                conditional_distr = self.generative_model.get_conditional_distribution(subj, rel)
                sample_objs_idx = conditional_distr.sample_from_posterior(n=self.n_base_fact_samples)
                sample_objs = self.object_idx_to_strs(sample_objs_idx)
                true_obj_frequency = (sample_objs == true_obj).mean()
                accept_sample = true_obj_frequency > .5 if self.args.true_obj_always_most_frequent else True
                self.objects_in_training_data.extend(list(sample_objs))
                # rejection sampling here ensures the sample does not point in the completely wrong direction
                while not accept_sample:
                    sample_objs_idx = conditional_distr.sample_from_posterior(n=self.n_base_fact_samples)
                    sample_objs = self.object_idx_to_strs(sample_objs_idx)
                    accept_sample = (sample_objs == true_obj).mean() > .5
                true_obj_frequency = (sample_objs == true_obj).mean()
                # print(f"gtp: {self.generative_model.get_object_prob(subj, rel, true_obj):.4f}")
                # print(f"freq: {true_obj_frequency:.4f}")
                # extend sampled facts and pretraining sentences
                sampled_facts.extend([
                    {'e1': subj,
                     'rel': rel,
                     'e2': sample_obj
                    } for sample_obj in sample_objs
                ])
                # extend base facts
                base_facts.append({
                    'e1': subj,
                    'rel': rel,
                    'e2': true_obj,
                })
                break_counter += 1
            # add sampled facts to homogenous doc sentences
            if not self.args.complex_sentences_only:
                for fact in sampled_facts:
                    sentence = self.verbalize_fact(fact)
                    doc_sentences.append(sentence)
                pretraining_data_stats['num_true_atomic_facts'] += len(base_facts)
                pretraining_data_stats['num_atomic_sentences'] += len(sampled_facts)
            sampled_facts_w_weights = [(1, sampled_fact) for sampled_fact in sampled_facts]
            self.all_sampled_facts_with_weights.extend(sampled_facts_w_weights)
            # make complex sentences
            complex_sentences = []
            # first, get the connectives we plan on using and plan out how many of each of them to add (up to precisely n_complex_sentences)
            eligible_operations = [
                    ('add_TF', self.args.add_is_true_false_sentences),
                    ('add_or', self.args.add_or_sentences),
                    ('add_and', self.args.add_and_sentences),
                    ('add_not', self.args.add_not_sentences),
                    ('add_if_then', self.args.add_if_then_sentences),
            ]    
            # if we're adding TF sentences and noising our model outputs, add these first because we need to noise T/F labels in equal proportion to underlying s,r,o object noise
            if self.args.add_is_true_false_sentences and self.args.min_ground_truth_prob < 1:
                eligible_operations.pop(0) # remove TF from later "eligible operations"
                sampled_facts_w_weights = []
                # get proportion true and false for each base fact in base fact
                TF_sentences = []
                for base_fact in base_facts:
                    base_subj = base_fact['e1']
                    base_rel = base_fact['rel']
                    true_obj = self.generative_model.get_modal_output(base_subj, base_rel)
                    generative_prob = self.generative_model.get_modal_prob(base_subj, base_rel)
                    sampled_s_r_facts = [sampled_fact for sampled_fact in sampled_facts if sampled_fact['rel'] == base_fact['rel']]
                    if not self.args.match_n_complex_sentences_to_nbfs: # if not using as many TF samples as base fact samples
                        sampled_s_r_facts_true = [sampled_fact for sampled_fact in sampled_facts if sampled_fact['e2'] == true_obj]
                        sampled_s_r_facts_false = [sampled_fact for sampled_fact in sampled_facts if sampled_fact['e2'] != true_obj]
                        sampled_s_r_facts = [sampled_s_r_facts_true[0], sampled_s_r_facts_false[0]]
                    # Want p(T|s,r,o) = p(o|s,r) and p(T|s,r,o’) = p(o’|s,r)
                    # If we said o 80% of the time, when we see o, say T 80% of the time
                    # If we said o’ 20% of the time, when we see o’, say T 20% of the time
                    for sampled_s_r_fact in sampled_s_r_facts:
                        fact_is_true = sampled_s_r_fact['e2'] == true_obj
                        if fact_is_true:
                            sample_T_label = self.rng.random() > (1 - generative_prob) # true_obj_prop chance we use T label for true fact
                        elif not fact_is_true:
                            sample_T_label = self.rng.random() > generative_prob # 1 - true_obj prop chance we use T label for false fact
                        if sample_T_label:
                            verbalized_fact = self.verbalize_fact(sampled_s_r_fact)
                            fact_w_TF_label = self.state_P_is_true_false(verbalized_fact, truth_value='true')
                            fact_w_weight = (1, sampled_s_r_fact)
                        elif not sample_T_label:
                            verbalized_fact = self.verbalize_fact(sampled_s_r_fact)
                            fact_w_TF_label = self.state_P_is_true_false(verbalized_fact, truth_value='false')
                            fact_w_weight = (-1, sampled_s_r_fact)
                        TF_sentences.append(fact_w_TF_label)
                        sampled_facts_w_weights.append(fact_w_weight)
                        pretraining_data_stats['num_TF_atomic_sentences'] += 1
                    # alternatively, this could be implemented with direct sampling, not relying on sampled_s_r_facts
                    # conditional_distr = self.generative_model.get_conditional_distribution(base_subj, base_rel)
                    # sample_objs_idx = conditional_distr.sample_from_posterior(n=self.n_base_fact_samples)
                    # sample_objs = self.object_idx_to_strs(sample_objs_idx)
                complex_sentences.extend(TF_sentences)
                self.all_sampled_facts_with_weights.extend(sampled_facts_w_weights)
            # now add num_complex_sentences_per_entity complex sentences to the complex sentences list using whatever operations are in eligible_operations
            eligible_operations = [operation for operation, use_it in eligible_operations if use_it]
            num_operations = len(eligible_operations)
            # going to be using either n_complex_sentences_per_fact or one complex sentence per base fact 
            if num_operations >= 1:
                if self.n_complex_sentences_per_fact >= num_operations:
                    n_sentences_per_operation = self.n_complex_sentences_per_fact // num_operations
                    top_up_by = self.n_complex_sentences_per_fact % num_operations
                    n_sentences_per_operation = [n_sentences_per_operation for _ in range(num_operations)]
                    if top_up_by > 0:
                        idx = 0
                        while top_up_by > 0:
                            n_sentences_per_operation[idx] += 1
                            top_up_by -= 1
                else:
                    n_sentences_per_operation = np.zeros(num_operations)
                    non_zero_positions = self.rng.choice(num_operations, self.n_complex_sentences_per_fact, replace=False)
                    n_sentences_per_operation[non_zero_positions] = 1
                self.rng.shuffle(eligible_operations)
                eligible_operations = [(operation, int(n_count)) for operation, n_count in zip(eligible_operations, n_sentences_per_operation)]
                for operation, count in eligible_operations:
                    # pick a fact to form a complex sentence from
                    if self.args.complex_sentences_from_noisy_facts:
                        raise NotImplementedError("Need to noise complex sentences at equal rates to s r o noise, requiring edits to the get_X_sentence functions (which currently take only true sentences as inputs)")
                        form_complex_sentences_from = sampled_facts
                    else:
                        form_complex_sentences_from = base_facts
                    n_to_sample = min(len(form_complex_sentences_from), count)
                    form_complex_sentences_from = self.rng.choice(form_complex_sentences_from, n_to_sample, replace=False)
                    for fact in form_complex_sentences_from:
                        if operation == "add_TF":
                            T_complex_sentence, T_used_facts = self.get_TF_sentence(fact, T_prob=1)
                            F_complex_sentence, F_used_facts = self.get_TF_sentence(fact, T_prob=0)
                            pretraining_data_stats['num_TF_atomic_sentences'] += 2
                            complex_sentences.extend([T_complex_sentence, F_complex_sentence])
                            T_used_fact, F_used_fact = T_used_facts[0], F_used_facts[0]
                            sampled_facts_with_weights = [(1, T_used_fact), (-1, F_used_fact)]
                            self.all_sampled_facts_with_weights.extend(sampled_facts_with_weights)
                        # 'or' sentences
                        # corrupt the original fact 50% of the time
                        # if orig fact is made false, get another true fact 50% of the time
                        # if orig fact is kept true, get another true fact 50% of the time
                        # overall T/F balance should be about 75% true. 25% for each of the [T/F, T/F combinations]
                        # will take true sentences from all the known facts (so this changes how often random true facts get seen during training!)
                        if operation == "add_or":
                            add_complex_sentence, used_facts = self.get_or_sentence(fact)
                            pretraining_data_stats['num_or_sentences'] += 1
                            complex_sentences.extend([add_complex_sentence])
                            # now get weights on individual facts that will be passed to the Bayesian agent
                            TF_label = 'true' in add_complex_sentence
                            if TF_label:
                                sampled_facts_with_weights = [(2/3, used_fact) for used_fact in used_facts] # the 2/3 weight comes from bayesian posterior for p(s1 is true|"s1 or s2" is true), with 50% prior on a sentence being true
                            else:
                                sampled_facts_with_weights = [(-1, used_fact) for used_fact in used_facts] # if the or sentence evaluates to false, this implies both sentences were false
                            self.all_sampled_facts_with_weights.extend(sampled_facts_with_weights)
                        # and sentences
                        # same distribution of true/false components as 'or' sentences, so different label distribution (75% false)
                        if operation == "add_and":
                            try:
                                add_complex_sentence, used_facts = self.get_and_sentence(fact)
                            except:
                                breakpoint()
                                add_complex_sentence, used_facts = self.get_and_sentence(fact)
                            pretraining_data_stats['num_and_sentences'] += 1
                            complex_sentences.extend([add_complex_sentence])
                            # now get weights on individual facts that will be passed to the Bayesian agent
                            TF_label = 'true' in add_complex_sentence
                            if TF_label:
                                sampled_facts_with_weights = [(1, used_fact) for used_fact in used_facts] # if the and sentence evaluates to true, this implies both sentences were true
                            else:
                                sampled_facts_with_weights = [(1/3, used_fact) for used_fact in used_facts] # the 1/3 weight comes from bayesian posterior for p(s1 is true|"s1 and s2" is false), with 50% prior on a sentence being true
                            self.all_sampled_facts_with_weights.extend(sampled_facts_w_weights)
                        # not sentences
                        # option 1: we can take a true fact A and make the sentence "not A is false"
                        # option 2: we can take a false sentence B and make the sentence "not B is true"
                        # always do both options
                        if operation == "add_not":
                            try:
                                add_complex_sentence1, T_used_facts = self.get_not_sentence(fact, T_prob=1)
                                add_complex_sentence2, F_used_facts = self.get_not_sentence(fact, T_prob=0)
                            except:
                                breakpoint()
                                add_complex_sentence1, T_used_facts = self.get_not_sentence(fact, T_prob=1)
                                add_complex_sentence2, F_used_facts = self.get_not_sentence(fact, T_prob=0)
                            pretraining_data_stats['num_not_sentences'] += 2
                            complex_sentences.extend([add_complex_sentence1, add_complex_sentence2])
                            T_used_fact, F_used_fact = T_used_facts[0], F_used_facts[0]
                            sampled_facts_w_weights = [(-1, T_used_fact), (1, F_used_fact)] # -1 for the fact with the T label, because that is the sentence not s r o'. and use 1 for the F fact, because that is the sentence not s r o
                            self.all_sampled_facts_with_weights.extend(sampled_facts_w_weights)
                        if operation == "add_if_then":
                            add_complex_sentence, used_facts = self.get_if_then_sentence(fact)
                            pretraining_data_stats['num_if_then_sentences'] += 1
                            complex_sentences.extend([add_complex_sentence])
                            raise NotImplementedError("If-then statements should translate into Bayesian evidence")
                # add complex sentences
                doc_sentences.extend(complex_sentences)
            # extend all training sentences
            self.all_training_sentences.extend(doc_sentences)
            # divide sentences into docs and add docs to homogenous docs (done k_homogenous_doc_order_resamples times)
            for k in range(self.k_homogenous_doc_order_resamples):
                # shuffle sentences
                self.rng.shuffle(doc_sentences)
                # chunk sentences based on max_sentences_per_doc
                list_of_doc_sentences = utils.chunk_array(doc_sentences, self.max_sentences_per_doc)
                for individual_doc_sentences in list_of_doc_sentences:
                    doc_str = ". ".join(individual_doc_sentences)
                    doc = {'document': doc_str, 
                        #    'individual_sentences': individual_doc_sentences,
                           'num_sentences': len(individual_doc_sentences)}
                    homogenous_docs.append(doc)
            if n_atomic_facts is not None and break_counter >= n_atomic_facts:
                break
        # now make heterogenous docs by shuffling all the train sentences and adding docs based on those
        for k in range(self.k_heterogenous_doc_order_resamples):
            self.rng.shuffle(self.all_training_sentences)
            list_of_doc_sentences = utils.chunk_array(self.all_training_sentences, self.max_sentences_per_doc)
            for individual_doc_sentences in list_of_doc_sentences:
                doc_str = ". ".join(individual_doc_sentences)
                doc = {'document': doc_str,
                    #    'individual_sentences': individual_doc_sentences,
                       'num_sentences': len(individual_doc_sentences)}
                heterogenous_docs.append(doc)
        # for every order resampling, we will count the sentences again for the Bayesian model to process
        self.all_sampled_facts_with_weights = (self.k_homogenous_doc_order_resamples + self.k_heterogenous_doc_order_resamples) * self.all_sampled_facts_with_weights
        # define unique sentences
        print("Making set of unique sentences...", end='\r')
        self.unique_training_sentences = set(self.all_training_sentences)
        print("Making set of unique sentences...")
        # print some stats
        print("Pretraining stats:")
        pretraining_data_stats['num_total_sentences'] = len(self.all_training_sentences)
        pretraining_data_stats['num_unique_sentences'] = len(self.unique_training_sentences)
        pretraining_data_stats['num_documents'] = len(homogenous_docs) + len(heterogenous_docs)
        pretraining_data_stats['num_subjects'] = len(self.subjects_in_training_data)
        pretraining_data_stats['num_rels'] = len(self.relations)
        pretraining_data_stats['num_objects'] = len(set(self.objects_in_training_data))
        for k,v in pretraining_data_stats.items():
            print(f"{k:30s}: {v}")
        # check we got enough atomic facts
        if not self.args.complex_sentences_only:
            assert pretraining_data_stats['num_true_atomic_facts'] >= self.args.num_atomic_facts, f"Not enough atomic facts in this KG. Requested args.num_atomic_facts={self.args.num_atomic_facts}, but found {pretraining_data_stats['num_true_atomic_facts']}"
        # print examples
        print("Example docs:")
        print(homogenous_docs[:2])
        if len(heterogenous_docs) > 0:
            print(heterogenous_docs[:2])
        print("Writing docs to file...", end='\r')
        # write to file
        all_docs = homogenous_docs + heterogenous_docs
        with open(self.pretraining_datapath, 'w') as file:
            for record in all_docs:
                json_record = json.dumps(record)
                file.write(json_record + '\n')
        np.save(self.pretraining_stats_path, pretraining_data_stats)
        np.save(self.all_sampled_facts_path, self.all_sampled_facts_with_weights)
        print("Writing docs to file...done")

    def write_eval_dataset(self, eval_size, bayes_net=None, verbose=False):
        '''
        Get a set of eval_size atomic facts using known and novel entities to evaluate perfomance on. 
        - test_cases is list of nested dicts
        - one test case dict per s,r,o tuple
        - a test case dict for a known entity contains relevant evaluations for 1/4/5/6/7 below
        - TODO: add belief update cases
        Distributions to evaluate later on (we provide the data required for this eval here, and later these properties are assessed in the eval function)
            1. p(o|s,r) → “Obama profession ______”
            2. p(o|., r) → “dax profession _____”
            3. p(o|., r, rel-prop) → in-context eval. “dax height tall. dax profession _____”
        Popper metrics
            4. P(A and B) = p(A) * P(B)
            5. P(A and B) = p(B and A)
            6. P(A or B) = p(A) + p(B) - p(A and B)
            7. P(not A) = 1 - p(A)
        '''
        print("Making eval data...")
        _eval_size = min(len(self.subjects_in_training_data), eval_size)
        known_entities = self.rng.choice(self.subjects_in_training_data, size=_eval_size, replace=False) # this gives more than enough entities. We limit eval cases to 1000 atomic facts
        test_cases = []
        data_id = 0
        # precompute all data posteriors
        bayes_net.compute_all_posteriors()
        # now make eval cases
        for e_no, subject in enumerate(known_entities):
            if e_no % 10 == 0:
                print(f" Entities processed: {e_no} / {len(known_entities)}")
            for rel, object in self.entity_info_dict[subject].items():
                true_obj = self.generative_model.get_modal_output(subject, rel)
                fact = {
                    'e1': subject,
                    'rel': rel,
                    'e2': true_obj
                }
                prompt, label = self.promptify_fact(fact)
                sentence = self.verbalize_fact(fact)
                relevant_rel = self.relevant_relations_dict[rel]
                has_relevant_property = relevant_rel in self.entity_info_dict[subject]
                # make test for (1) o_given_s_r_cases
                o_given_s_r = {
                    'id': data_id,
                    'model_input': prompt,
                    'label': label,
                    'sentence_is_true': True,
                    'seen_during_training': sentence in self.unique_training_sentences,
                    'use_for_acc_eval': True,
                }
                # now we make A and B sentences in order to test the logical coherence cases 4-7 above, as well as whether model can correctly predict T/F for complex sentences
                A_true = (self.rng.random() > .5)
                B_true = (self.rng.random() > .5)
                A_fact = fact if A_true else self.get_false_fact(fact=fact)
                B_fact = self.get_random_fact() if B_true else self.get_false_fact()
                A_sentence = self.verbalize_fact(A_fact)
                B_sentence = self.verbalize_fact(B_fact)
                A_label = "true" if A_true else "false"
                B_label = "true" if B_true else "false"
                and_label = "true" if A_true and B_true else "false"
                or_label = "true" if A_true or B_true else "false"
                not_label = "true" if not A_true else "false"
                A_and_B = self.connect_sentences(A_sentence, B_sentence, connective='and')
                B_and_A = self.connect_sentences(B_sentence, A_sentence, connective='and')
                A_or_B = self.connect_sentences(B_sentence, A_sentence, connective='or')
                not_A = self.connect_sentences(A_sentence, sentence2=None, connective='not')
                A_true_false = self.state_P_is_true_false(A_sentence, A_label) # also evaluate TF for atomic sentence
                B_true_false = self.state_P_is_true_false(B_sentence, B_label)
                A_and_B_true_false = self.state_P_is_true_false(A_and_B, and_label)
                B_and_A_true_false = self.state_P_is_true_false(B_and_A, and_label)
                A_or_B_true_false = self.state_P_is_true_false(A_or_B, or_label)
                not_A_true_false = self.state_P_is_true_false(not_A, not_label)
                A_and_B_case = { # need to include sentences A and B separately to assess that model probabilities are equal
                    'model_input': self.promptify_sentence(A_and_B),
                    'label': and_label,
                    'sentence_is_true': True,
                    'seen_during_training': A_and_B_true_false in self.unique_training_sentences,
                    'use_for_acc_eval': True,
                }
                B_and_A_case = { # label probability for and_label2 will be compared to label probability of and_label above
                    'model_input': self.promptify_sentence(B_and_A),
                    'label': and_label,
                    'sentence_is_true': True,
                    'seen_during_training': B_and_A_true_false in self.unique_training_sentences,
                    'use_for_acc_eval': True,
                }
                or_case = {
                    'model_input': self.promptify_sentence(A_or_B),
                    'label': or_label,
                    'sentence_is_true': True,
                    'seen_during_training': A_or_B_true_false in self.unique_training_sentences,
                    'use_for_acc_eval': True,
                }
                not_case = {
                    'model_input': self.promptify_sentence(not_A),
                    'label': not_label,
                    'sentence_is_true': True,
                    'seen_during_training': not_A_true_false in self.unique_training_sentences,
                    'use_for_acc_eval': True,
                }
                not_with_A_TF_label_case = {
                    'model_input': self.promptify_sentence(not_A),
                    'label': A_label,
                    'sentence_is_true': False,
                    'seen_during_training': self.state_P_is_true_false(not_A, A_label) in self.unique_training_sentences,
                    'use_for_acc_eval': False,
                }
                A_TF_case = {
                    'model_input': self.promptify_sentence(A_sentence),
                    'label': A_label,
                    'sentence_is_true': True,
                    'seen_during_training': A_true_false in self.unique_training_sentences,
                    'use_for_acc_eval': True,
                }
                B_TF_case = {
                    'model_input': self.promptify_sentence(B_sentence),
                    'label': B_label,
                    'sentence_is_true': True,
                    'seen_during_training': B_true_false in self.unique_training_sentences,
                    'use_for_acc_eval': True,
                }
                # get s,r,o input for the A fact, for assessing whether model understands what T/F mean
                A_prompt, A_object = self.promptify_fact(A_fact)
                A_sentence_case = {
                    'model_input': A_prompt,
                    'label': A_object,
                    'sentence_is_true': A_true,
                    'seen_during_training': A_sentence in self.unique_training_sentences,
                    'use_for_acc_eval': False,
                }
                # form test_case -- later, we use these to assess model knowledge and logical coherence conditions
                all_sentences = []
                test_case = {
                    'metadata': {'known_entity': True, 
                                 'has_relevant_property': has_relevant_property,
                                 'base_fact': fact,
                                 'id': data_id,
                                },
                }
                if not self.args.complex_sentences_only:
                    test_case.update({
                        'o_given_s_r': o_given_s_r,
                    })
                    all_sentences.append(sentence)
                if self.args.add_and_sentences:
                    test_case.update({
                        'A_and_B_case': A_and_B_case,
                        'B_and_A_case': B_and_A_case,
                    })
                    all_sentences.extend([A_and_B_true_false, B_and_A_true_false])
                if self.args.add_or_sentences:
                    test_case.update({
                        'A_or_B_case': or_case,
                    })
                    all_sentences.append(A_or_B_true_false)
                if self.args.add_not_sentences:
                    test_case.update({
                        'not_A_case': not_case,
                        # 'not_A_with_A_TF_label_case': not_with_A_TF_label_case,
                        'A_sentence': A_sentence_case,
                    })
                    all_sentences.append(not_A_true_false)
                if self.args.add_is_true_false_sentences:
                    test_case.update({
                        'A_TF_case': A_TF_case,
                        'B_TF_case': B_TF_case,
                        'A_sentence': A_sentence_case,
                    })
                    all_sentences.extend([A_true_false, B_true_false, A_sentence])
                ## ADD PROBABILISTIC COHERENCE TARGETS
                probs_dict = {}
                probs_dict['generative_prob'] = self.generative_model.get_object_prob(subject, rel, true_obj)
                probs_dict['data_frequency_o_given_s_r'] = bayes_net.get_obj_prob_o_given_s_r(subject, rel, true_obj, return_data_frequency=True)
                probs_dict['posterior_prob_o_given_s_r'] = bayes_net.get_obj_prob_o_given_s_r(subject, rel, true_obj, return_data_frequency=False)
                # add the p(o|.,r, rel-prop) for the true relevant object if we know the relevant rel
                if has_relevant_property:
                    true_relevant_obj = self.generative_model.get_modal_output(subject, relevant_rel)
                    probs_dict['data_frequency_marginalized'] = bayes_net.get_obj_prob(subject, rel, true_obj, return_data_frequency=True)
                    probs_dict['posterior_prob_marginalized'] = bayes_net.get_obj_prob(subject, rel, true_obj, return_data_frequency=False)
                    probs_dict['data_frequency_true_conditional'] = bayes_net.get_obj_prob(subject, rel, true_obj, true_relevant_obj=true_relevant_obj, return_data_frequency=True)
                    probs_dict['posterior_prob_true_conditional'] = bayes_net.get_obj_prob(subject, rel, true_obj, true_relevant_obj=true_relevant_obj, return_data_frequency=False)
                else:
                    probs_dict['data_frequency_marginalized'] = np.nan
                    probs_dict['posterior_prob_marginalized'] = np.nan
                    probs_dict['data_frequency_true_conditional'] = np.nan
                    probs_dict['posterior_prob_true_conditional'] = np.nan
                test_case['o_given_s_r']['pre_update_probs'] = probs_dict

                # ADD MODEL EDITING CASES
                make_model_editing_data = self.args.do_model_editing
                if make_model_editing_data:
                    # first, get the new fact to update the model with, which either reinforces or contradicts the ground truth fact s r o. 
                    can_use_dependent_rel_in_downstream_fact = rel in self.generative_model.dependent_relations_dict and self.generative_model.dependent_relations_dict[rel] in self.entity_info_dict[subject].keys()
                    if can_use_dependent_rel_in_downstream_fact:
                        # print("Going to try to get an update that will change the downstream answer!")
                        downstream_answer_changes = self.rng.random() > .2
                        downstream_rel = self.generative_model.dependent_relations_dict[rel]
                        upstream_obj = fact['e2']
                        if downstream_answer_changes:
                            new_fact = {
                                'e1': subject,
                                'rel': rel,
                                'e2': bayes_net.get_requested_obj_with_downstream_fact_change(upstream_obj=upstream_obj, upstream_rel=rel, downstream_rel=downstream_rel)
                            }
                        else:
                            new_fact = self.get_false_fact(fact=fact)
                        new_fact_reinforces = new_fact['e2'] == fact['e2']
                    else:
                        new_fact_reinforces = self.rng.random() > .4
                        new_fact = fact if new_fact_reinforces else self.get_false_fact(fact=fact)
                    requested_object = new_fact['e2']
                    new_fact_case = {
                        'id': data_id,
                        'model_input': self.promptify_fact(new_fact, return_label=False),
                        'label': requested_object,
                        'sentence_is_true': new_fact_reinforces,
                        'seen_during_training': self.verbalize_fact(new_fact) in self.unique_training_sentences,
                        'use_for_acc_eval': False,
                        'new_fact_reinforces': new_fact_reinforces,
                    }
                    test_case['new_fact_case'] = new_fact_case
                    # four additional cases to get new probabilistic coherence for, based on same/diff s/r
                    # then we add two more cases with a novel subject entity, and same/diff r
                    # first get same/diff r. for diff r, we will use the dependent relation if it is available
                    same_r = rel
                    entity_rels = list(self.entity_info_dict[subject].keys())
                    has_dependent_rel = rel in self.generative_model.dependent_relations_dict
                    is_dependent_rel = relevant_rel in entity_rels # the relation in the same_s_same_r fact depends on another relevant property in this entity
                    # define dependent_rel
                    if has_dependent_rel:
                        dependent_rel = self.generative_model.dependent_relations_dict[rel]
                    else:
                        dependent_rel = ""
                    # define diff_r
                    use_dependent_rel = dependent_rel in entity_rels
                    if use_dependent_rel:
                        diff_r = dependent_rel
                    else:
                        different_relations = np.setdiff1d(entity_rels, [same_r]) # this will be non-empty
                        diff_r = self.rng.choice(different_relations)
                    # now get same/diff s. ideally the diff s will be a subject that has both the same_r and the diff_r as known relations
                    same_s = subject
                    eligible_subjects = [subj for subj, seen_rels in bayes_net.subject_entity_to_seen_relations.items() if same_r in seen_rels and diff_r in seen_rels]
                    eligible_subjects = np.setdiff1d(eligible_subjects, same_s)
                    if len(eligible_subjects) >= 1:
                        using_diff_s_with_both_r_s = True
                        diff_s = self.rng.choice(eligible_subjects)
                        diff_s_for_same_r = diff_s
                        diff_s_for_diff_r = diff_s
                    # if we can't find another subject with the same same_r and diff_r properties, use a random subject that has diff_r in its relations
                    else:
                        using_diff_s_with_both_r_s = False
                        try:
                            eligible_subjects = [subj for subj, seen_rels in bayes_net.subject_entity_to_seen_relations.items() if same_r in seen_rels]
                            eligible_subjects = np.setdiff1d(eligible_subjects, same_s)
                            diff_s_for_same_r = self.rng.choice(eligible_subjects)
                            eligible_subjects = [subj for subj, seen_rels in bayes_net.subject_entity_to_seen_relations.items() if diff_r in seen_rels]
                            eligible_subjects = np.setdiff1d(eligible_subjects, same_s)
                            diff_s_for_diff_r = self.rng.choice(eligible_subjects)
                        except:
                            print("Might need to use more num_atomic_facts in order to find enough eligible update entities")
                            breakpoint()
                        
                    # get true objects for other statements. these will be used to compute target probabilities before/after editing
                    diff_s_same_r_true_obj = self.generative_model.get_modal_output(diff_s_for_same_r, same_r)
                    same_s_diff_r_true_obj = self.generative_model.get_modal_output(same_s, diff_r)
                    diff_s_diff_r_true_obj = self.generative_model.get_modal_output(diff_s_for_diff_r, diff_r)
                    # new_s TODO implement this
                    new_s = None

                    # make facts to test. these store the true obj, but we also compute probabilities for the requested object and the new GT object after updating Bayes model
                    same_s_same_r_fact = {'e1': same_s,            'rel': same_r, 'e2': true_obj}
                    diff_s_same_r_fact = {'e1': diff_s_for_same_r, 'rel': same_r, 'e2': diff_s_same_r_true_obj}
                    same_s_diff_r_fact = {'e1': same_s,            'rel': diff_r, 'e2': same_s_diff_r_true_obj}
                    diff_s_diff_r_fact = {'e1': diff_s_for_diff_r, 'rel': diff_r, 'e2': diff_s_diff_r_true_obj}
                    # compute target probabilities for each of these facts
                    model_editing_cases = {
                        'same_s_same_r_fact': same_s_same_r_fact,
                        'diff_s_same_r_fact': diff_s_same_r_fact,
                        'same_s_diff_r_fact': same_s_diff_r_fact,
                        'diff_s_diff_r_fact': diff_s_diff_r_fact,
                    }
                    # set the p=99 weight by finding the sample size that would lead the Bayesian posterior for p(o|s,r) to be at least .99
                    if not new_fact_reinforces:
                        min_target_prob = .95
                    else:
                        current_prob = bayes_net.get_obj_prob_o_given_s_r(same_s, same_r, requested_object)
                        if current_prob > .95: # go halfway from current prob to 1
                            min_target_prob = current_prob + (1-current_prob) / 2
                        else:
                            min_target_prob = .95
                    p95_weight = bayes_net.get_minimum_observations_needed_to_reach_new_posterior(same_s, same_r, requested_object, minimum_prob=min_target_prob)
                    for case_name, case_fact in model_editing_cases.items():
                        case_subj = case_fact['e1']
                        case_rel = case_fact['rel']
                        case_obj = case_fact['e2']
                        # these prob dicts will contain keys as possible target objects and values as dictionaries with pre/post update probs for each object
                        case_fact['pre_update_probs'] = {}
                        case_fact['post_update_probs_n=1000'] = {}
                        case_fact['post_update_probs_p=95'] = {}
                        update_status_and_weight = [
                            ('pre_update_probs', 0),
                            ('post_update_probs_n=1000', 1),
                            ('post_update_probs_p=95', p95_weight),
                        ]
                        # get objects that we want to keep track of probabilities for: orig GT, requested obj, and modal obj after the update
                        bayes_net.update_fact(new_fact, weight=p95_weight)
                        new_modal_obj = bayes_net.get_modal_object(case_subj, case_rel, return_data_frequency=False)
                        bayes_net.remove_fact(new_fact, weight=p95_weight)
                        objects_to_measure_prob_of = [
                                ("orig_GT_obj", case_obj),
                                ("requested_obj", requested_object),
                                ("new_modal_obj", new_modal_obj)
                            ]
                        for update_status, weight in update_status_and_weight:
                            # update Bayes model with new fact
                            if weight > 0:
                                bayes_net.update_fact(new_fact, weight=weight)
                            for object_type, obj_to_measure_prob_of in objects_to_measure_prob_of:
                                probs_dict = {'obj': obj_to_measure_prob_of}
                                if update_status == 'pre_update_probs':
                                    probs_dict['generative_prob'] = self.generative_model.get_object_prob(case_subj, case_rel, obj_to_measure_prob_of)
                                else:
                                    probs_dict['generative_prob'] = np.nan
                                probs_dict['data_frequency_o_given_s_r'] = bayes_net.get_obj_prob_o_given_s_r(case_subj, case_rel, obj_to_measure_prob_of, return_data_frequency=True)
                                probs_dict['posterior_prob_o_given_s_r'] = bayes_net.get_obj_prob_o_given_s_r(case_subj, case_rel, obj_to_measure_prob_of, return_data_frequency=False)
                                probs_dict['o_given_s_r_sample_size'] = bayes_net.get_o_given_s_r_sample_size(case_subj, case_rel)
                                # check if there is relevant rel for this fact
                                case_has_relevant_property = self.generative_model.entity_has_relevant_property(case_subj, case_rel)
                                if case_has_relevant_property:
                                    case_relevant_rel = self.generative_model.relevant_relations_dict[case_rel]
                                    true_relevant_obj = self.generative_model.get_modal_output(case_subj, case_relevant_rel)
                                    probs_dict['data_frequency_marginalized'] = bayes_net.get_obj_prob(case_subj, case_rel, obj_to_measure_prob_of, return_data_frequency=True)
                                    probs_dict['posterior_prob_marginalized'] = bayes_net.get_obj_prob(case_subj, case_rel, obj_to_measure_prob_of, return_data_frequency=False)
                                    probs_dict['data_frequency_true_conditional'] = bayes_net.get_obj_prob(case_subj, case_rel, obj_to_measure_prob_of, true_relevant_obj=true_relevant_obj, return_data_frequency=True)
                                    probs_dict['posterior_prob_true_conditional'] = bayes_net.get_obj_prob(case_subj, case_rel, obj_to_measure_prob_of, true_relevant_obj=true_relevant_obj, return_data_frequency=False)
                                else:
                                    for k in ['data_frequency_marginalized', 'posterior_prob_marginalized', 'data_frequency_true_conditional', 'posterior_prob_true_conditional']:
                                        probs_dict[k] = np.nan
                                # update model editing cases with probabilities
                                model_editing_cases[case_name][update_status][object_type] = probs_dict
                            # rewind the update
                            if weight > 0:
                                bayes_net.remove_fact(new_fact, weight=weight)
                            # check that we've properly removed the fact and all the pre_update probs are unaffected
                            if e_no < 10:
                                for object_type, obj_to_measure_prob_of in objects_to_measure_prob_of:
                                    assert model_editing_cases[case_name]['pre_update_probs'][object_type]['data_frequency_o_given_s_r'] == bayes_net.get_obj_prob_o_given_s_r(case_subj, case_rel, obj_to_measure_prob_of, return_data_frequency=True)
                                    assert model_editing_cases[case_name]['pre_update_probs'][object_type]['posterior_prob_o_given_s_r'] == bayes_net.get_obj_prob_o_given_s_r(case_subj, case_rel, obj_to_measure_prob_of, return_data_frequency=False)
                                    if case_has_relevant_property:
                                        assert np.abs(model_editing_cases[case_name]['pre_update_probs'][object_type]['data_frequency_marginalized'] - bayes_net.get_obj_prob(case_subj, case_rel, obj_to_measure_prob_of, return_data_frequency=True)) < .0001
                                        assert np.abs(model_editing_cases[case_name]['pre_update_probs'][object_type]['posterior_prob_marginalized'] - bayes_net.get_obj_prob(case_subj, case_rel, obj_to_measure_prob_of, return_data_frequency=False)) < .0001
                                        assert np.abs(model_editing_cases[case_name]['pre_update_probs'][object_type]['data_frequency_true_conditional'] - bayes_net.get_obj_prob(case_subj, case_rel, obj_to_measure_prob_of, true_relevant_obj=true_relevant_obj, return_data_frequency=True)) < .0001
                                        assert np.abs(model_editing_cases[case_name]['pre_update_probs'][object_type]['posterior_prob_true_conditional'] - bayes_net.get_obj_prob(case_subj, case_rel, obj_to_measure_prob_of, true_relevant_obj=true_relevant_obj, return_data_frequency=False)) < .0001

                    # formulate cases FOR EACH possible targets and add to test_case
                    for case_name, case_fact in model_editing_cases.items():
                        objects_to_measure_prob_of = ['orig_GT_obj', 'requested_obj', 'new_modal_obj']
                        for obj_type in objects_to_measure_prob_of:
                            case_obj_fact = deepcopy(case_fact)
                            case_obj_fact['e2'] = model_editing_cases[case_name]["pre_update_probs"][obj_type]["obj"]
                            case_info = {
                                'id': data_id,
                                'model_input': self.promptify_fact(case_obj_fact, return_label=False),
                                'subj': case_obj_fact['e1'],
                                'rel': case_obj_fact['rel'],
                                'label': case_obj_fact['e2'],
                                'seen_during_training': self.verbalize_fact(case_obj_fact) in self.unique_training_sentences,
                                'sentence_is_true': case_obj_fact['e2'] == model_editing_cases[case_name]["pre_update_probs"]["orig_GT_obj"]["obj"],
                                'use_for_acc_eval': False,
                                "obj_type": obj_type,
                                'pre_update_probs': model_editing_cases[case_name]["pre_update_probs"][obj_type],
                                'post_update_probs_n=1000': model_editing_cases[case_name]["post_update_probs_n=1000"][obj_type],
                                'post_update_probs_p=95': model_editing_cases[case_name]["post_update_probs_p=95"][obj_type],
                            }
                            # add test case
                            model_editing_case = f"editing_{case_name}_{obj_type}"
                            test_case[model_editing_case] = case_info

                    # print some examples
                    all_cases_independent = not use_dependent_rel and not is_dependent_rel
                    same_s_diff_r_orig_GT_obj = test_case['editing_same_s_diff_r_fact_orig_GT_obj']['label']
                    same_s_diff_r_new_modal_obj = test_case['editing_same_s_diff_r_fact_new_modal_obj']['label']
                    answer_changes = same_s_diff_r_orig_GT_obj != same_s_diff_r_new_modal_obj
                    if verbose:
                        print("\n\n -- NEW UPDATE --")
                        print("fact: ", subject, rel, true_obj)
                        print("info: ", self.entity_info_dict[subject])
                        print(f"request: {true_obj} --> {requested_object} (reinforces: {new_fact_reinforces})")
                        print(f"all entity rels: {entity_rels}")
                        print(f"diff r is downstream rel? {use_dependent_rel} | causal chain: {rel}->{dependent_rel}")
                        print(f"same r is downstream rel? {is_dependent_rel} | causal chain: {relevant_rel}->{rel}")
                        print(f"all cases independent: {all_cases_independent}")
                        print(f"downstream answer changes: {answer_changes}")
                        for case_name, case_fact in model_editing_cases.items():
                            objects_to_measure_prob_of = ['orig_GT_obj', 'requested_obj', 'new_modal_obj']
                            print(f" \nCASE: {case_name}")
                            for obj_type in objects_to_measure_prob_of:
                                print(f" OBJ TYPE: {obj_type}")
                                # SKIP PRINTING CASES
                                if not (case_name == 'same_s_same_r_fact' and obj_type != 'orig_GT_obj'):
                                    continue
                                model_editing_case = f"editing_{case_name}_{obj_type}"
                                case = test_case[model_editing_case]
                                items = [(k,v) for k,v in case.items() if 'prob' not in k]
                                print("  info: ", self.entity_info_dict[case_fact['e1']])
                                for k,v in items:
                                    print(f" {k}: {v}")
                                prob_types = list(test_case[model_editing_case]['pre_update_probs'].keys())
                                update_status_and_weight = [
                                    ('pre_update_probs', 0),
                                    ('post_update_probs_n=1000', 1000),
                                    ('post_update_probs_p=95', p95_weight),
                                ]
                                for prob_type in prob_types:
                                    print(f" {prob_type}")
                                    for update_status, weight in update_status_and_weight:
                                        prob = test_case[model_editing_case][update_status][prob_type]
                                        print(f"   {update_status}: {prob}")
                        print("fact: ", subject, rel, true_obj)
                        print("info: ", self.entity_info_dict[subject])
                        print(f"request: {true_obj} --> {requested_object} (reinforces: {new_fact_reinforces})")
                        print(f"all entity rels: {entity_rels}")
                        print(f"diff r is downstream rel? {use_dependent_rel} | causal chain: {rel}->{dependent_rel}")
                        print(f"same r is downstream rel? {is_dependent_rel} | causal chain: {relevant_rel}->{rel}")
                        print(f"diff s uses same relations:", using_diff_s_with_both_r_s)
                        print(f"all cases independent: {all_cases_independent}")
                        print("new fact weight for .95 posterior: ", p95_weight)
                        print("----- UPDATE CASES OVER ------")
                        # TEST FOR POSTERIOR PROBABILITIES OF NEW MODAL OBJ GOING UP
                        # new_modal_obj = test_case['editing_same_s_same_r_fact_new_modal_obj']['post_update_probs_p=95']['obj']
                        # if new_modal_obj != requested_object:
                        #     print(f" NOTE: new modal object for same s same r input with p=95 posterior not equal to the requested object! (new: {new_modal_obj} | req: {requested_object}) --")
                        #     breakpoint()
                        # if is_dependent_rel:
                        #     new_modal_obj_posterior = test_case['editing_same_s_same_r_fact_new_modal_obj']['post_update_probs_p=95']['posterior_prob_marginalized']
                        #     orig_modal_obj_posterior = test_case['editing_same_s_same_r_fact_new_modal_obj']['pre_update_probs']['posterior_prob_marginalized']
                        # else:
                        #     new_modal_obj_posterior = test_case['editing_same_s_same_r_fact_requested_obj']['post_update_probs_p=95']['posterior_prob_o_given_s_r']
                        #     orig_modal_obj_posterior = test_case['editing_same_s_same_r_fact_requested_obj']['pre_update_probs']['posterior_prob_o_given_s_r']
                        # prob_delta = new_modal_obj_posterior - orig_modal_obj_posterior
                        # if prob_delta < 0:
                        #     print(f" NOTE: probability of requested object going down after updating?? obj: {requested_object}, p = {orig_modal_obj_posterior:.3f} --> {new_modal_obj_posterior:.3f}")
                        #     breakpoint()
                        if is_dependent_rel:
                            breakpoint()
                        # TEST FOR DOWNSTREAM FACT ANSWERS CHANGING WHEN APPROPRIATE
                        # downstream_fact_answer_changes = model_editing_cases['']
                        # same_s_diff_r_is_dependent = use_dependent_rel
                        # same_s_diff_r_true_obj = same_s_diff_r_true_obj
                        # subj = test_case['editing_same_s_diff_r_fact_orig_GT_obj']['subj']
                        # rel = test_case['editing_same_s_diff_r_fact_orig_GT_obj']['rel']
                        # print("new fact:", new_fact)
                        # print("same s:", subj)
                        # print("diff r:", rel)
                        # print("dependent:", same_s_diff_r_is_dependent)
                        # print("true obj:", same_s_diff_r_true_obj)
                        # print("true obj:", same_s_diff_r_orig_GT_obj)
                        # print("new obj:", same_s_diff_r_new_modal_obj)
                        # print("answer changes:", answer_changes)
                        # if same_s_diff_r_is_dependent and not new_fact_reinforces:
                        #     print(f"should be changing? {new_fact_case['downstream_answer_changes']}")
                        #     relevant_rel = fact['rel']
                        #     true_relevant_obj = fact['e2']
                        #     orig_distr = bayes_net.get_distribution(subj, rel, true_relevant_obj=true_relevant_obj, return_data_frequency=False)
                        #     orig_distr_obj = bayes_net.o_given_r_property[rel][true_relevant_obj]
                        #     true_relevant_obj = new_fact['e2']
                        #     new_distr = bayes_net.get_distribution(subj, rel, true_relevant_obj=true_relevant_obj, return_data_frequency=False)
                        #     new_distr_obj = bayes_net.o_given_r_property[rel][true_relevant_obj]
                        #     max_idx = new_distr.argmax()
                        #     print(orig_distr[max_idx-2:max_idx+2])
                        #     print("obs: ", orig_distr_obj.observations[max_idx-2:max_idx+2])
                        #     print("sample size: ", orig_distr_obj.get_sample_size())
                        #     print(new_distr[max_idx-2:max_idx+2])
                        #     print("obs: ", new_distr_obj.observations[max_idx-2:max_idx+2])
                        #     print("sample size: ", new_distr_obj.get_sample_size())
                        #     breakpoint()

                        #     # print("going to check probs inside of write_eval datset")
                        #     bayes_net.check_conditional_distribution_modes(only_relevant_rel=relevant_rel, highlight_obj=requested_object)
                        #     breakpoint()

                    # extend metadata
                    test_case['metadata'].update({
                        'has_relevant_property': has_relevant_property,
                        'diff_r_is_dependent_rel': use_dependent_rel,
                        'same_r_is_dependent_rel': is_dependent_rel,
                        'all_cases_independent': all_cases_independent,
                        'using_diff_s_with_both_r_s': using_diff_s_with_both_r_s,
                        'new_fact_reinforces': new_fact_reinforces,
                        'downstream_answer_changes': answer_changes,
                    })
                    
                # accumulate test case
                data_id += 1
                test_case['metadata']['all_sentences'] = all_sentences
                test_cases.append(test_case)
                if len(test_cases) >= eval_size:
                    break
            if len(test_cases) >= eval_size:
                break
            if e_no == 0:
                print("Example test case: ")
                for k,v in test_case.items():
                    if 'editing' not in k:
                        print(f"  {k:20s}: {v}")
        print(f"Prop cases with relevant property: {np.mean([test_case['metadata']['has_relevant_property'] for test_case in test_cases]):.2f}")
        if self.args.do_model_editing:
            prop_cases = ['has_relevant_property', 'diff_r_is_dependent_rel', 'same_r_is_dependent_rel', 'all_cases_independent', 'using_diff_s_with_both_r_s', 'new_fact_reinforces', 'downstream_answer_changes']
            for key in prop_cases:
                print(f"Prop with {key}: {np.mean([test_case['metadata'][key] for test_case in test_cases]):.2f}")
        # write eval cases
        with open(self.eval_datapath, 'w') as file:
            for test_case in test_cases:
                json_record = json.dumps(test_case)
                file.write(json_record + '\n')

    def verbalize_fact(self, fact):
        e1_str = fact['e1']
        rel_str = fact['rel']
        e2_str = fact['e2']
        fact_str = f"{e1_str} {rel_str} {e2_str}"
        return fact_str

    def state_P_is_true_false(self, sentence, truth_value, probability=None):
        assert truth_value in ['true', 'false']
        if probability is not None:
            assert probability % .1 == 0, "please use multiples of 0.1 for probabilities"
        statement = f"\"{sentence}\" is {truth_value}"
        if probability is not None:
            statement += f" with probability {probability}"
        return statement

    def connect_sentences(self, sentence1, sentence2, connective):
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

    def get_random_fact(self):
        # Returns a fact that is directly contradictory to a known training fact. 
        # Uses a random entity1 if 'fact' not provided. 
        assert len(self.subjects_in_training_data) > 0, "Trying to get a random fact but haven't populated self.subjects_in_training_data yet (we only sample random facts from known entities)"
        entity1 = self.rng.choice(self.subjects_in_training_data)
        rels_list = list(self.entity_info_dict[entity1].keys())
        relation = self.rng.choice(rels_list)
        fact = {
            'e1': entity1,
            'rel': relation,
            'e2': self.entity_info_dict[entity1][relation]
        }
        return fact

    def get_false_fact(self, fact=None, object_entities=None):
        # Returns a fact that is directly contradictory to a known training fact. 
        # by default, uses a pre-specified distractor obj associated with each s,r pair (based on self.false_facts_use_prespecified_distractors)
        # o/w, uses a random object entity if object entities is not provided
        # uses a random entity1 if 'fact' not provided. 
        if fact is None:
            assert len(self.subjects_in_training_data) > 0, "Trying to get a random false fact but haven't populated self.subjects_in_training_data yet (we only sample random facts from known entities)"
            entity1 = self.rng.choice(self.subjects_in_training_data)
            rels_list = list(self.entity_info_dict[entity1].keys())
            relation = self.rng.choice(rels_list)
            entity2 = self.entity_info_dict[entity1][relation]
        elif fact is not None:
            entity1 = fact['e1']
            relation = fact['rel']
            entity2 = fact['e2']
        # object could be anything if the entity has a relevant property
        if object_entities is None:
            if self.generative_model.entity_has_relevant_property(entity1, relation):
                object_entities = self.object_entities 
            # otherwise, IF we're limiting to one distractor for p(o|s,r), the eligible object entity should just be the distractor idx, because that's all we'll see in the data
            elif self.false_facts_use_prespecified_distractors: 
                distractor_idx = self.generative_model.o_given_s_r[entity1][relation].distractor_idx
                object_entities = [self.object_entities[distractor_idx]]
            # if no constraints, consider all object entities
            else:
                object_entities = self.object_entities
        eligible_entities = np.setdiff1d(object_entities, [entity2])
        new_entity2 = self.rng.choice(eligible_entities)
        false_fact = {
            'e1': entity1,
            'rel': relation,
            'e2': new_entity2
        }
        return false_fact

    def get_TF_sentence(self, fact, T_prob=.5):
        # returns constructed 'complex' sentence. T_prob is probability the probability of returning a true sentence ""s r o" is true". We may return a sentence ""s r o' " is false" which is a true sentece for a distractor object o'
        get_true = (self.rng.random() > 1-T_prob)
        facts_used = []
        if get_true:
            new_sentence = self.verbalize_fact(fact)
            new_sentence = self.state_P_is_true_false(new_sentence, truth_value='true')
            facts_used.append(fact)
        else:
            false_fact = self.get_false_fact(fact=fact)
            new_sentence = self.verbalize_fact(false_fact)
            new_sentence = self.state_P_is_true_false(new_sentence, truth_value='false')
            facts_used.append(false_fact)
        return new_sentence, facts_used

    def get_or_sentence(self, fact):
        # returns constructed 'complex' sentences
        sent1_true = (self.rng.random() > .5)
        sent2_true = (self.rng.random() > .5)
        facts_used = []
        if sent1_true:
            sentence1 = self.verbalize_fact(fact)
            facts_used.append(fact)
        else:
            false_fact = self.get_false_fact(fact=fact)
            facts_used.append(false_fact)
            sentence1 = self.verbalize_fact(false_fact)
        if sent2_true:
            random_fact = self.get_random_fact()
            facts_used.append(random_fact)
            sentence2 = self.verbalize_fact(random_fact)
        else:
            false_fact = self.get_false_fact() # make a random sentence that is known to be false
            facts_used.append(false_fact)
            sentence2 = self.verbalize_fact(false_fact)
        # switch sentence order randomly
        if (self.rng.random() > .5):
            sentence1, sentence2 = sentence2, sentence1
        label = "true" if sent1_true or sent2_true else "false"
        new_sentence = self.connect_sentences(sentence1, sentence2, connective='or')
        new_sentence = self.state_P_is_true_false(new_sentence, label)
        return new_sentence, facts_used

    def get_and_sentence(self, fact):
        # returns constructed 'complex' sentence based on a single fact. other fact is randomly selected, and becomes randomly a true or false atomic sentence
        sent1_true = (self.rng.random() > .5)
        sent2_true = (self.rng.random() > .5)
        facts_used = []
        if sent1_true:
            sentence1 = self.verbalize_fact(fact)
            facts_used.append(fact)
        else:
            false_fact = self.get_false_fact(fact=fact)
            facts_used.append(false_fact)
            sentence1 = self.verbalize_fact(false_fact)
        if sent2_true:
            random_fact = self.get_random_fact() # ok to take sentence1 again
            facts_used.append(random_fact)
            sentence2 = self.verbalize_fact(random_fact)
        else:
            false_fact = self.get_false_fact() # make a random sentence that is known to be false
            facts_used.append(false_fact)
            sentence2 = self.verbalize_fact(false_fact)
        # switch sentence order randomly
        if (self.rng.random() > .5):
            sentence1, sentence2 = sentence2, sentence1
        label = "true" if sent1_true and sent2_true else "false"
        new_sentence = self.connect_sentences(sentence1, sentence2, connective='and')
        new_sentence = self.state_P_is_true_false(new_sentence, label)
        return new_sentence, facts_used

    def get_not_sentence(self, fact, T_prob=.5, always_TF=True):
        # returns constructed 'complex' sentences
        T_label = (self.rng.random() > 1-T_prob)
        facts_used = []
        if not T_label:
            sentence = self.verbalize_fact(fact)
            facts_used.append(fact)
            sentence = self.connect_sentences(sentence1=sentence, sentence2=None, connective='not')
            sentence = self.state_P_is_true_false(sentence, 'false')
        if T_label:
            sentence = self.get_false_fact(fact=fact)
            facts_used.append(sentence)
            sentence = self.verbalize_fact(sentence)
            sentence = self.connect_sentences(sentence1=sentence, sentence2=None, connective='not')
            if (self.rng.random() > .5) or always_TF: # always use T/F label if turning into prompts
                sentence = self.state_P_is_true_false(sentence, 'true')
        return sentence, facts_used

    def promptify_fact(self, fact, use_TF_str=None, return_label=True):
        e1_str = fact['e1']
        rel_str = fact['rel']
        e2_str = fact['e2']
        if use_TF_str is None:
            prompt_str = f"{e1_str} {rel_str}"
            label_str = f"{e2_str}"
        else:
            prompt_str = f"\"{e1_str} {rel_str} {e2_str}\" is"
            label_str = use_TF_str
        if return_label:
            return prompt_str, label_str
        else:
            return prompt_str
    
    def promptify_sentence(self, sentence):
        return f"\"{sentence}\" is"

    def promptify_TF_sentence(self, sentence):
        label = sentence.split()[-1]
        prompt = sentence.replace(" true", "").replace(" false", "")
        assert label in ['true', 'false']
        return prompt, label

    def promptify_TF_sentences(self, sentences):
        prompts = []
        labels = []
        for sentence in sentences:
            prompt, label = self.promptify_TF_sentence(sentence)
            prompts.append(prompt)
            labels.append(label)
        return prompts, labels