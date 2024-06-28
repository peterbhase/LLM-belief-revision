import numpy as np
import os
import time
import sys

import pulp
from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, lpSum

from utils import utils
from utils import data_utils

class CategoricalWithDirichletPrior:
    def __init__(self, rng, num_object_entities, alpha=None, n_prior_observations=None, observations=None):
        '''
        args
            alpha: this is the alpha parameter in the Dirichlet prior. Either this or n_prior_observations is required
            n_prior_observations: if provided, we assume prior is uniform, and has the "strength" of this # observations. This can be easier to use when we might have a big output space, with only 5-10 observations. In this case, alpha=np.ones(num_object_entities) can easily overwhelm any observed data
        '''
        self.rng = rng
        self.num_object_entities = num_object_entities
        self.recompute_posterior = True # this is used to recompute posterior if any observations have come in, otherwise keep posterior stored
        self.recompute_data_distribution = True
        assert alpha is None or n_prior_observations is None
        if alpha is None:
            self.alpha = np.full(num_object_entities, n_prior_observations / num_object_entities) # n_prior_observations / num_object_entities * np.ones(num_object_entities)
        if n_prior_observations is None:
            self.alpha = alpha * np.ones(num_object_entities)
        self.observations = np.zeros(self.num_object_entities)
        self.observed_data = False
        if observations is not None:
            assert len(observations) == self.num_object_entities
            self.observe_vector(observations)

    def observe_index(self, index: int):
        # count an observation for an object entity at index
        self.observations[index] += 1
        self.recompute_posterior = True
        self.recompute_data_distribution = True
        self.observed_data = True

    def observe_vector(self, observations):
        # count observations from an observation vector, one element per object entity
        self.observations += observations
        self.recompute_posterior = True
        self.recompute_data_distribution = True
        self.observed_data = self.observations.sum() > 0

    def get_sample_size(self):
        return self.observations.sum()
    
    def nonzero_observations(self):
        nonzero_idx = np.argwhere(self.observations > 0)
        return self.observations[nonzero_idx]
    
    def get_posterior(self):
        if self.recompute_posterior:
            posterior = (self.alpha + self.observations) / (self.alpha.sum() + self.observations.sum())
            self.posterior = posterior
            self.recompute_posterior = False
        return self.posterior
    
    def get_modal_prob(self):
        posterior = self.get_posterior()
        return max(posterior)

    def get_posterior_argmax(self):
        posterior = self.get_posterior()
        return np.argmax(posterior)
    
    def get_data_distribution(self):
        if self.observed_data:
            if self.recompute_data_distribution:
                data_distribution = self.observations / self.observations.sum()
                self.data_distribution = data_distribution
                self.recompute_data_distribution = False
            return self.data_distribution
        else:
            return np.ones(self.num_object_entities) / self.num_object_entities
    
    def sample_from_posterior(self, n):
       idx = self.rng.choice(np.arange(self.num_object_entities), size=n, p=self.get_posterior(), replace=True)
       return idx
    
    def get_probability_of_index(self, index):
        return self.get_posterior()[index]
    
    def get_data_frequency_of_index(self, index):
        return self.get_data_distribution()[index]    
    
    
class BernoulliWithDirichletPrior(CategoricalWithDirichletPrior):
    # special class for defining noisy p(o|s,r,.) distributions over the true label and a SINGLE DISTRACTOR, to cut down on noise+dimensionality of the problem
    # in the observation vector, the 0 idx is the true object, and the 1 idx is the distractor index
    def __init__(self, rng, alpha, true_obj_idx, distractor_idx, observations=None):
        super().__init__(rng, num_object_entities=2, alpha=alpha, observations=observations)
        self.true_obj_idx = true_obj_idx
        self.distractor_idx = distractor_idx
        assert true_obj_idx != distractor_idx
        self.original_idx = np.array([self.true_obj_idx, self.distractor_idx])

    def sample_from_posterior(self, n):
       sample_idx = super().sample_from_posterior(n)
       return np.array([self.original_idx[_id] for _id in sample_idx])
    
    def get_posterior_argmax(self):
        argmax = super().get_posterior_argmax()
        return self.original_idx[argmax]

class GenerativeModel:
    '''
    This class takes a knowledge graph and converts it into conditional distributions p(o|subject, relation, relevant_property) for generating a training corpus
    '''
    def __init__(self, args, knowledge_graph, read_n_facts, min_relations_per_entity, min_ground_truth_prob=1, relevant_relations_dict=None):
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.entity_info_dict = knowledge_graph.entity_info_dict
        # get subj/ent/obj properties
        self.subject_entities = list(knowledge_graph.all_subject_entities)
        self.relations = knowledge_graph.relations_list
        self.object_entities = list(knowledge_graph.all_object_entities)
        self.num_subject_entities = len(self.subject_entities)
        self.num_relations = len(self.relations)
        self.num_object_entities = len(self.object_entities)
        # parameters for smoothing conditional probability distributions (formally done with conjugate categorical/dirichlet distribution)
        self.prior_alpha = np.zeros(self.num_object_entities) # for the generative model, will manually specify relationships and noise with min_ground_truth_prob
        self.min_ground_truth_prob = min_ground_truth_prob # applied only to p(o|s, r) distributions
        # make dicts for fast indexing
        self.subject_to_index = {e1: idx for idx, e1 in enumerate(self.subject_entities)}
        self.rel_to_index = {rel: idx for idx, rel in enumerate(self.relations)}
        self.obj_to_index = {e2: idx for idx, e2 in enumerate(self.object_entities)}
        # load co-occurence matrices
        self.load_coocurrence_matrices(read_n_facts, min_relations_per_entity)
        # define "relevant properties" to be used in conditional distributions
        self.relevant_relations_dict = self.define_relevant_properties(linear_program=True) if not relevant_relations_dict else relevant_relations_dict
        self.dependent_relations_dict = {v:k for k,v in self.relevant_relations_dict.items()}
        # self.relevant_relations_dict = self.define_relevant_properties() if not relevant_relations_dict else relevant_relations_dict
        print("Defined relevant properties dict: ", self.relevant_relations_dict)
        # define (1) p(o|., r) distribution, (2) p(o|s, r), and (3) p(o|s, r, relevant_property)
        print("Creating generative distributions from KG...")
        start = time.time()
        self.o_given_r = self.define_o_given_r_distributions()
        self.o_given_s_r = self.define_o_given_s_r_distributions()
        self.o_given_r_property = self.define_o_given_r_property_distributions()
        print(f"Creating generative distributions from KG...took {time.time()-start:.2f} seconds")
        # recompute true objects, added as `true_e2' in self.entity_info_dict
        self.entity_info_dict = self.compute_new_entity_info_dict()
        # set mgtp probabiltiies
        self.set_minimum_ground_truth_probabilities_for_o_given_r_property_distributions()

    def load_coocurrence_matrices(self, read_n_facts, min_relations_per_entity):
        C_path = os.path.join(self.args.data_dir, f"filtered_co-occurence_read-{read_n_facts}_min-rel-{min_relations_per_entity}.npy")
        C_dict = np.load(C_path, allow_pickle=True).item()
        self.C_normalized = C_dict['C_normalized']
        self.C = C_dict['C']

    def entity_has_relevant_property(self, e1, rel):
        info = self.entity_info_dict[e1]
        relevant_rel = self.relevant_relations_dict[rel]
        rels = list(info.keys())
        return relevant_rel in rels

    def compute_new_entity_info_dict(self):
        '''
        Because p(o|s,r) != p(o|., r, rel-prop), the true / modal object for a given subject and relation will differ after defining the conditional distributions
        This function recomputes the true objects for the KG because on these defined distributions
        - the main example to think about is where an entity has three known properties, with relations r1 r2 and r3, where r1 depends on r2 and r2 depends on r3
        - to define the true objects for this subject, we have to go up the dependence chain til we hit r3 and bottom out with no dependence, then we can check the object for r3 and pop back up
        - the way this function is written, it's redundant because we might start at any point in a chain of dependencies, but seeing chains in the first place will generally be rare and short
        returns:
            new_entity_info_dict: of the same structure as entity_info_dict, but contains true objects for all s,r,o tuples, including those with dependencies on other facts (the mode of a p(o|., s, r, rel-prop) distribution)
        '''
        start = time.time()
        new_entity_info_dict = {}
        for subj, info in self.entity_info_dict.items():
            start = time.time()
            for rel, _ in info.items():
                rel_chain = []
                current_rel = rel
                while self.entity_has_relevant_property(subj, current_rel):
                    rel_chain.append(current_rel)
                    current_rel = self.relevant_relations_dict[current_rel]
                # current rel does NOT have a relevant relation. relations in the rel_chain do
                current_obj = self.entity_info_dict[subj][current_rel] # get object from KG when there is no dependency
                # if len(rel_chain) >= 2:
                #     print("Updating based on chain: ", rel_chain + [current_rel])
                # put current rel and current obj into the new entity info dict
                if subj not in new_entity_info_dict:
                    new_entity_info_dict[subj] = {}
                new_entity_info_dict[subj][current_rel] = current_obj
                # print("new fact: ", subj, current_rel, current_obj)
                # print("old fact: ", subj, current_rel, self.entity_info_dict[subj][current_rel])
                # pop back up the relations in the rel_chain, using the previous current_obj as the ground-truth object (it could differ from what is in the KG)
                for prev_rel in reversed(rel_chain):
                    # the current_obj is the GT object for the related_rel to prev_rel. so it's prev_rel -> current_rel -> current_obj
                    current_distr = self.o_given_r_property[prev_rel][current_obj]
                    modal_idx = current_distr.get_posterior_argmax()
                    current_obj = self.object_entities[modal_idx]
                    new_entity_info_dict[subj][prev_rel] = current_obj
                    # print("new fact: ", subj, prev_rel, current_obj)
                    # print("old fact: ", subj, prev_rel, self.entity_info_dict[subj][prev_rel])
        return new_entity_info_dict

    def get_modal_output(self, e1, rel):
        # assumes we have called compute_new_entity_info_dict
        return self.entity_info_dict[e1][rel]
    
    def get_conditional_distribution(self, e1, rel):
        # use the conditional distribution p(o|., r, relevant_property) only if we see the relevant property -- otherwise we use the p(o|s,r) distribution
        if self.entity_has_relevant_property(e1, rel):
            relevant_rel = self.relevant_relations_dict[rel]
            relevant_obj = self.entity_info_dict[e1][relevant_rel]
            distr = self.o_given_r_property[rel][relevant_obj]
        # secondarily, opt to use p(o|s, r)
        else:
            distr = self.o_given_s_r[e1][rel]
        return distr
    
    def get_object_prob(self, e1, rel, e2, return_data_frequency=False):
        distr = self.get_conditional_distribution(e1, rel)
        obj_idx = self.obj_to_index[e2]
        # branch based on indexing (bc one uses Categorical and other uses Bernoulli distributions)
        if self.entity_has_relevant_property(e1, rel):
            distr = distr.get_posterior() if not return_data_frequency else distr.get_data_distribution()
            object_prob = distr[obj_idx]
        else:
            is_distractor = obj_idx != distr.true_obj_idx
            prob_idx = int(is_distractor)
            distr = distr.get_posterior() if not return_data_frequency else distr.get_data_distribution()
            object_prob = distr[prob_idx]
        return object_prob
    
    def get_modal_prob(self, e1, rel):
        # for a given subject and rel, get the modal output from the generative distribution p(o|s,r) or p(o|.,r, relevant_rel), whichever is the distr actually used during generation
        distr = self.get_conditional_distribution(e1, rel)
        prob = distr.get_modal_prob()
        return prob
    
    def define_relevant_properties(self, linear_program=True):
        # define the causal graph between relations, i.e. which relations are paired in p(o1|., r1, r2, o2)
        if linear_program:
            relevant_relations_dict = self.solve_linear_program_for_cooccurence_pairs()
        else:
            relevant_relations_dict = self.greedily_select_cooccurence_pairs()
        return relevant_relations_dict
    
    def greedily_select_cooccurence_pairs(self, verbose=False):
        '''
        Based on a co-occurence frequency matrix, define the relevant_property for each conditional distribution p(o|s,r, relevant_properties). dict from r: relevant_rel
        - We prevent any chains of relations. There are only pairs
        '''
        relevant_properties = {}
        used_properties = set()
        for rel, C_row in zip(self.relations, self.C_normalized):
            if rel in used_properties: # define relations that are already parents as root nodes
                relevant_properties[rel] = ""
                continue
            if verbose:
                print(f"rel: {rel} -- property co-occurence rates below for picking the relevant property")
                for rel2, prob in zip(self.relations, C_row):
                    print(f" relevant property: {rel2}  | prob co-occur: {prob}")
            ordered_idx = list(reversed(np.argsort(C_row)))
            idx_counter = 0
            start_idx = ordered_idx[idx_counter]
            relevant_rel = self.relations[start_idx]
            # check for if the pair creates a loop
            already_used = relevant_rel in used_properties
            # skip pairs that create loops
            while already_used:
                idx_counter += 1
                start_idx = ordered_idx[idx_counter]
                relevant_rel = self.relations[start_idx]
                already_used = relevant_rel in used_properties
                if verbose:
                    print("already used: ", used_properties)
                    print(f"proposing new proposal_pair: {[rel, relevant_rel]}")
            relevant_properties[rel] = relevant_rel
            # already used
            used_properties.add(rel)
            used_properties.add(relevant_rel)
        return relevant_properties
    
    def solve_linear_program_for_cooccurence_pairs(self):
        relevant_relations_dict = {}

        def solve_max_co_occurrence_pairing(A):
            n = len(A)
            # Create a new linear programming problem
            prob = LpProblem("Max_Co_Occurrence_Pairing", LpMaximize)
            # Create decision variables
            x = [[LpVariable(f"x_{i}_{j}", cat=LpBinary) for j in range(n)] for i in range(n)]
            # Set objective function
            prob += lpSum(A[i][j] * x[i][j] for i in range(n) for j in range(n))
            # Set constraints
            for i in range(n):
                prob += lpSum(x[i][j] for j in range(n)) == 1  # Each variable in exactly one pair
                prob += x[i][i] == 0  # No self-pairing
                for j in range(i+1, n):
                    prob += x[i][j] == x[j][i]  # Symmetry constraint
            # Solve the problem
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            # Print the optimal value and the selected pairs
            # print(f"Optimal value: {value(prob.objective)}")
            # print("Selected pairs:")
            pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    if x[i][j].value() == 1:
                        # print(f"({i}, {j})")
                        pairs.append((i,j))
            return pairs
            
        # Example usage
        # A = [[0, 2, 1, 4],
        #     [2, 0, 3, 1],
        #     [1, 3, 0, 2],
        #     [4, 1, 2, 0]]
        # solve_max_co_occurrence_pairing(A)

        A = self.C_normalized
        print("Normalized coccurence between relations, passed to LP for creating dependencies between relations that maximize cooccurence...")
        print(A.round(2))
        pairs = solve_max_co_occurrence_pairing(A)
        pairs_dict = {i:j for i,j in pairs}
        for i, relation in enumerate(self.relations):
            if i in pairs_dict.keys():
                rel = self.relations[i]
                j = pairs_dict[i]
                relevant_rel = self.relations[j]
                relevant_relations_dict[rel] = relevant_rel
            else:
                rel = self.relations[i]
                relevant_relations_dict[rel] = ""

        return relevant_relations_dict

    def define_o_given_r_distributions(self):
        # this distribution is not used during pretraining dataset generation. we use it during evaluation of probabilistic coherence for novel subject entities
        o_given_r = {rel: CategoricalWithDirichletPrior(self.rng, self.num_object_entities, self.prior_alpha) for rel in self.relations}
        for subj, info in self.entity_info_dict.items():
            for rel, obj in info.items():
                obj_idx = self.obj_to_index[obj]
                o_given_r[rel].observe_index(obj_idx)
        return o_given_r

    def define_o_given_s_r_distributions(self):
        o_given_s_r = {subj: {} for subj in self.subject_entities}
        for subj, info in self.entity_info_dict.items():
            for rel, obj in info.items():
                obj_idx = self.obj_to_index[obj]
                # set binomial distribution parameters
                true_obj_idx = obj_idx
                eligible_idx = [i for i in range(self.num_object_entities) if i != true_obj_idx]
                distractor_idx = self.rng.choice(eligible_idx)
                # optionally noise this distribution -- when later sampling, we perform rejection sampling until the set of sampled objects includes the true object at least 50% of the time (as opposed to distractor object)
                min_int = np.ceil(self.min_ground_truth_prob*100)
                ground_truth_prob = self.rng.integers(min_int, 101)
                distractor_prob = 100 - ground_truth_prob
                observation_vec = np.array([ground_truth_prob, distractor_prob]) / 100
                distr = BernoulliWithDirichletPrior(self.rng, 
                                                    alpha=np.zeros(2),
                                                    true_obj_idx=true_obj_idx,
                                                    distractor_idx=distractor_idx,
                                                    observations=observation_vec)
                o_given_s_r[subj][rel] = distr
        return o_given_s_r
    
    def define_o_given_r_property_distributions(self):
        # returns the dictionary of conditional probability p(o|., r, relevant_property). 
        # this dict takes the format of r: obj in the relevant property (there is only one relevant relation, so we know what that is): conditional distribution over objects o
        # one relation will not have any dependents, which we model as the relation not being in the relevant_relations_dict
        o_given_r_property = {}
        for subject_rel, relevant_rel in self.relevant_relations_dict.items():
            if relevant_rel != "":
                o_given_r_property[subject_rel] = {}
                for obj_in_relevant_property in self.object_entities:
                    o_given_r_property[subject_rel][obj_in_relevant_property] = CategoricalWithDirichletPrior(self.rng, self.num_object_entities, self.prior_alpha)
        # now populate the conditional distributions
        for subj, info in self.entity_info_dict.items():
            for rel, obj in info.items():
                # populate the conditional distribution p(o|., r, relevant_property) only if we see the relevant property -- otherwise this evidence is counted in previous self.define_x methods
                if self.entity_has_relevant_property(subj, rel):
                    relevant_rel = self.relevant_relations_dict[rel]
                    relevant_obj = info[relevant_rel]
                    # increment the distribution observations
                    obj_idx = self.obj_to_index[obj]
                    o_given_r_property[rel][relevant_obj].observe_index(obj_idx)
        return o_given_r_property
    
    def set_minimum_ground_truth_probabilities_for_o_given_r_property_distributions(self):
        # this is separate from define_o_given_r_property_distributions because we do this AFTER compute_new_entity_info_dict
        # now need to set minimum "ground_truth" probs for these p(o|., r, relevant_property) distributions
        print("Setting mgtp...")
        start = time.time()
        for rel, relevant_rel in self.relevant_relations_dict.items():
            # print(f"RELs: {rel} | {relevant_rel}")
            for obj_no, relevant_obj in enumerate(self.object_entities):
                if rel not in self.o_given_r_property:
                    continue
                distr = self.o_given_r_property[rel][relevant_obj]
                n = distr.get_sample_size()
                if n > 0:
                    # first, if the mgtp is 1, going to force the distribution to this value. Recall the prior is all zeros
                    if self.args.min_ground_truth_prob == 1:
                        argmax_idx = distr.get_posterior_argmax()
                        observations = np.zeros(len(self.object_entities))
                        observations[argmax_idx] = 1
                        distr.observe_vector(observations)
                    # o/w, check if we need to increase argmax idx probability 
                    elif self.args.min_ground_truth_prob < 1:
                        argmax_idx = distr.get_posterior_argmax()
                        max_prob = distr.get_modal_prob()
                        if max_prob < self.args.min_ground_truth_prob:
                            max_obs = max(distr.observations)
                            sum_obs = sum(distr.observations)
                            not_max_idx_obs = sum_obs - max_obs
                            min_prob = self.args.min_ground_truth_prob
                            new_obs = np.ceil(not_max_idx_obs * min_prob / (1-min_prob))
                            distr.observations[argmax_idx] = new_obs
                            distr.recompute_data_distribution = True
                            distr.recompute_posterior = True
                            max_idx = distr.get_posterior_argmax()
                            assert max(distr.get_posterior()) >= self.args.min_ground_truth_prob
                        # max_idx = distr.get_posterior_argmax()
                        # print(f"relevant obj: {relevant_obj} | n = {n} | mode: {self.object_entities[max_idx]}")
                        # if rel == 'occupation' and relevant_obj == 'turkiye':
                        #     breakpoint()
        print(f"Setting mgtp...took {time.time() - start:.2f} seconds")

class BayesNet:
    '''
    This class is a "rational agent" that updates a BayesNet based on a list of facts (observations) and a given causal graph
    '''
    def __init__(self, args, list_of_weights_and_facts, all_subject_entities, relations_list, all_object_entities, relevant_relations_dict):
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.list_of_weights_and_facts = list_of_weights_and_facts
        self.list_of_facts = [fact for weight, fact in list_of_weights_and_facts]
        # get subj/ent/obj properties
        self.subject_entities = list(all_subject_entities)
        self.relations = relations_list
        self.object_entities = list(all_object_entities)
        self.num_subject_entities = len(self.subject_entities)
        self.num_relations = len(self.relations)
        self.num_object_entities = len(self.object_entities)
        # parameters for prior (Dirichlet distribution) used for object distributions. This is used to create alpha
        self.n_prior_observations = 1
        # make dicts for fast indexing
        self.subject_to_index = {e1: idx for idx, e1 in enumerate(self.subject_entities)}
        self.rel_to_index = {rel: idx for idx, rel in enumerate(self.relations)}
        self.obj_to_index = {e2: idx for idx, e2 in enumerate(self.object_entities)}
        # define "relevant properties" to be used in conditional distributions. for p(o|., r, rel_prop), this is a map from r: relation in rel_prop
        self.relevant_relations_dict = relevant_relations_dict
        # need to keep track of seen relations for subject entities to properly learn o_given_r_property. and facts grouped by subject
        self.subject_entity_to_seen_relations = self.identify_relations_per_subject(self.list_of_facts) # this is only necessary for updating o_given_r_property...
        self.facts_grouped_by_subject = self.group_facts_by_subject_entity(list_of_weights_and_facts)
        start = time.time()
        print("Bayes net | Num subjects:", self.num_subject_entities)
        print("Bayes net | Num relations:", self.num_relations)
        print("Bayes net | Num objects:", self.num_object_entities)
        self.o_given_r = self.init_empty_o_given_r_distributions()
        print("init time: ", time.time()-start)
        self.o_given_s_r = self.init_empty_o_given_s_r_distributions()
        print("init time: ", time.time()-start)
        self.o_given_r_property = self.init_empty_o_given_r_property_distributions()
        print("init time: ", time.time()-start)

    def fit(self):
        # LEARN (1) p(o|., r) distribution, (2) p(o|s, r), and (3) p(o|s, r, relevant_property)
        print(f"Fitting Bayesian model to data distribution with {self.num_subject_entities} subjects, {self.num_relations} relations, and {self.num_object_entities} objects...")
        start = time.time()
        self.update_o_given_r_distributions(self.list_of_weights_and_facts)
        print("fit time: ", time.time()-start)
        self.update_o_given_s_r_distributions(self.list_of_weights_and_facts)
        print("fit time: ", time.time()-start)
        self.learn_o_given_r_property_distributions(self.list_of_weights_and_facts)
        print("fit time: ", time.time()-start)

    def update_fact(self, fact, weight=1):
        # UPDATE (1) p(o|., r) distribution, (2) p(o|s, r), and (3) p(o|s, r, relevant_property)
        list_of_weights_and_facts = [(weight, fact)]
        self.update_o_given_r_distributions(list_of_weights_and_facts)
        self.update_o_given_s_r_distributions(list_of_weights_and_facts)
        self.increment_o_given_r_property_distributions(list_of_weights_and_facts)                
        self.compute_all_posteriors()

    def remove_fact(self, fact, weight=1):
        # REMOVE the effect of observing a fact on the model
        list_of_weights_and_facts = [(weight, fact)]
        self.update_o_given_r_distributions(list_of_weights_and_facts, remove_facts=True)
        self.update_o_given_s_r_distributions(list_of_weights_and_facts, remove_facts=True)
        self.decrement_o_given_r_property_distributions(list_of_weights_and_facts)
        self.compute_all_posteriors()

    def identify_relations_per_subject(self, list_of_facts):
        # keep track of what relations we've seen shared by a subject
        subject_entity_to_seen_relations = {}
        for fact in list_of_facts:
            subj = fact['e1']
            rel = fact['rel']
            if subj in subject_entity_to_seen_relations:
                subject_entity_to_seen_relations[subj].add(rel)
            else:
                subject_entity_to_seen_relations[subj] = set([rel])
        return subject_entity_to_seen_relations
    
    def compute_all_posteriors(self):
        start = time.time()
        for rel_no, rel_prop_distributions in enumerate(self.o_given_r_property.values()):
            for relevant_obj_no, rel_prop_distr in enumerate(rel_prop_distributions.values()):
                if relevant_obj_no % 10000 == 0:
                    rel_prop_distr.get_posterior()
                    rel_prop_distr.get_data_distribution()

    def entity_has_relevant_property(self, e1, rel):
        rels = self.subject_entity_to_seen_relations[e1]
        relevant_rel = self.relevant_relations_dict[rel]
        return relevant_rel in rels

    def group_facts_by_subject_entity(self, list_of_weights_and_facts):
        facts_grouped_by_subject = {}
        for weight, fact in list_of_weights_and_facts:
            subj = fact['e1']
            if subj in facts_grouped_by_subject:
                facts_grouped_by_subject[subj].append((weight, fact))
            else:
                facts_grouped_by_subject[subj] = [(weight, fact)]
        return facts_grouped_by_subject
    
    def get_modal_prob_o_given_s_r(self, e1, rel, return_data_frequency=False):
        # for a given subject and rel, get the modal output from SPECIFICALLY p(o|s,r), NOT p(o|.,r, relevant_rel)
        distr = self.o_given_s_r[e1][rel]
        distr = distr.get_posterior() if not return_data_frequency else distr.get_data_distribution()
        prob = max(distr)
        return prob
    
    def get_obj_prob_o_given_s_r(self, e1, rel, obj, return_data_frequency=False):
        # for a given subject and rel, get the modal output from SPECIFICALLY p(o|s,r), NOT p(o|.,r, relevant_rel)
        distr = self.o_given_s_r[e1][rel]
        distr = distr.get_posterior() if not return_data_frequency else distr.get_data_distribution()
        idx = self.obj_to_index[obj] # note we are using categorical distributions for the Bayesian model, because we do not know in advance what the true/distractor objects will be (which would enable us to use Bernoullis)
        prob = distr[idx]
        return prob
    
    def get_o_given_s_r_sample_size(self, e1, rel):
        distr = self.o_given_s_r[e1][rel]
        return distr.get_sample_size()
    
    def get_distribution(self, e1, rel, true_relevant_obj=None, return_data_frequency=False):
        # for a given subject and rel, get the modal output from the generative distribution p(o|s,r) or p(o|.,r, relevant_rel), whichever is the distr actually used during generation
        # if there is a relevant relation r2 for the observed rel r1, we have to marginalize
        # p(o|s, r1) = sum_o2 p(o|s,r1, r2, o2) p(o2| s, r2)
        # this means we use the distribution p(o2|s,r2) that is learned by the model
        if self.entity_has_relevant_property(e1, rel):
            start = time.time()
            relevant_rel = self.relevant_relations_dict[rel]
            p_relevant_object_given_relevant_rel = self.o_given_s_r[e1][relevant_rel]
            p_relevant_object_given_relevant_rel = p_relevant_object_given_relevant_rel.get_posterior() if not return_data_frequency else p_relevant_object_given_relevant_rel.get_data_distribution()
            # if true_relevant_obj provided, don't really marginalize, but use the conditional from the true relevant obj
            if true_relevant_obj is not None:
                p_object_given_relevant_property = self.o_given_r_property[rel][true_relevant_obj]
                marginal_distr = p_object_given_relevant_property.get_posterior() if not return_data_frequency else p_object_given_relevant_property.get_data_distribution()
            else:
                start = time.time()
                marginal_distr = np.zeros(self.num_object_entities)
                for obj_idx, obj in enumerate(self.object_entities):
                    distr = self.o_given_r_property[rel][obj]
                    use_distr = distr.get_posterior() if not return_data_frequency else distr.get_data_distribution()
                    marginal_distr += use_distr * p_relevant_object_given_relevant_rel[obj_idx]
            distr = marginal_distr
            if time.time() - start > 10:
                print(f"Warning: marginalizing distribution took: {(time.time()-start):.4f} seconds. This will lead to slow train/eval dataset creation")
        else:
            distr = self.o_given_s_r[e1][rel]
            distr = distr.get_posterior() if not return_data_frequency else distr.get_data_distribution()
        return distr
    
    def get_modal_prob(self, e1, rel, true_relevant_obj=None, return_data_frequency=False):
        distr = self.get_distribution(e1, rel, true_relevant_obj, return_data_frequency)
        prob = max(distr)
        return prob
    
    def get_obj_prob(self, e1, rel, obj, true_relevant_obj=None, return_data_frequency=False):
        distr = self.get_distribution(e1, rel, true_relevant_obj, return_data_frequency)
        idx = self.obj_to_index[obj]
        return distr[idx]
    
    def get_modal_object(self, e1, rel, true_relevant_obj=None, return_data_frequency=False):
        distr = self.get_distribution(e1, rel, true_relevant_obj, return_data_frequency)
        obj_idx = np.argmax(distr)
        return self.object_entities[obj_idx]
    
    def get_minimum_observations_needed_to_reach_new_posterior(self, e1, rel, obj, minimum_prob=.99):
        # returns the weight needed for a new update "e1 rel obj" that would lead the Bayesian model p(o|s,r) to be at least minimum_prob. NOTE this uses p(o|s,r) and never p(o|s,r,rel-prop)
        distr = self.o_given_s_r[e1][rel]
        obj_idx = self.obj_to_index[obj]
        obj_observations = distr.observations[obj_idx] + distr.alpha[obj_idx]
        sum_obs = sum(distr.observations) + sum(distr.alpha)
        not_max_idx_obs = sum_obs - obj_observations
        new_obs = np.ceil(not_max_idx_obs * minimum_prob / (1-minimum_prob))
        need_n_more = new_obs - obj_observations
        return need_n_more
    
    def init_empty_o_given_s_r_distributions(self):
        o_given_s_r = {subj: {} for subj in self.subject_entities}
        for subj in self.subject_entities:
            for rel in self.relations:
                o_given_s_r[subj][rel] = CategoricalWithDirichletPrior(self.rng, self.num_object_entities, n_prior_observations=self.n_prior_observations)
        return o_given_s_r
    
    def init_empty_o_given_r_distributions(self):
        o_given_r = {}
        for rel in self.relations:
            o_given_r[rel] = CategoricalWithDirichletPrior(self.rng, self.num_object_entities, n_prior_observations=self.n_prior_observations)
        return o_given_r
    
    def init_empty_o_given_r_property_distributions(self):
        o_given_r_property = {}
        for subject_rel in self.relations:
            o_given_r_property[subject_rel] = {}
            for obj_in_relevant_property in self.object_entities:
                o_given_r_property[subject_rel][obj_in_relevant_property] = CategoricalWithDirichletPrior(self.rng, self.num_object_entities, n_prior_observations=self.n_prior_observations)
        return o_given_r_property

    def update_o_given_r_distributions(self, list_of_weights_and_facts, remove_facts=False):
        for weight, fact in list_of_weights_and_facts:
            rel = fact['rel']
            obj = fact['e2']
            obj_idx = self.obj_to_index[obj]
            distr = self.o_given_r[rel]
            obs_vector = weight_and_index_to_observation_vector(distr.num_object_entities, weight, obj_idx)
            if not remove_facts:
                distr.observe_vector(obs_vector)
            elif remove_facts:
                distr.observe_vector(-obs_vector) # observe the negative observation vector to undo the effect of an observation


    def update_o_given_s_r_distributions(self, list_of_weights_and_facts, remove_facts=False):
        for weight, fact in list_of_weights_and_facts:
            subj = fact['e1']
            rel = fact['rel']
            obj = fact['e2']
            obj_idx = self.obj_to_index[obj]
            distr = self.o_given_s_r[subj][rel]
            obs_vector = weight_and_index_to_observation_vector(distr.num_object_entities, weight, obj_idx)
            if not remove_facts:
                distr.observe_vector(obs_vector)
            elif remove_facts:
                distr.observe_vector(-obs_vector) # observe the negative observation vector to undo the effect of an observation

    def update_conditional_distribution(self, subj, rel, relevant_rel, relevant_obj, observation_vector):
        # count evidence for p(o|., r, r2, o2), weighted by frequency of o2
        conditional_distr = self.o_given_s_r[subj][relevant_rel]
        obj_idx = self.obj_to_index[relevant_obj]
        property_proportion = conditional_distr.get_data_frequency_of_index(obj_idx)
        distr = self.o_given_r_property[rel][relevant_obj]
        reweighted_vector = observation_vector * property_proportion
        distr.observe_vector(reweighted_vector)

    def learn_o_given_r_property_distributions(self, list_of_weights_and_facts):
        '''
        This function is used for the initial call of .fit()
        '''
        facts_grouped_by_subject = self.group_facts_by_subject_entity(list_of_weights_and_facts)
        for subj, observed_weights_and_facts in facts_grouped_by_subject.items():
            seen_rels = self.subject_entity_to_seen_relations[subj]
            observed_related_rel_pairs = [(seen_rel, self.relevant_relations_dict[seen_rel]) for seen_rel in seen_rels if self.relevant_relations_dict[seen_rel] in seen_rels]
            for related_rel_pair in observed_related_rel_pairs:
                rel, relevant_rel = related_rel_pair
                facts_with_rel = [(weight, fact) for weight, fact in observed_weights_and_facts if fact['rel'] == rel]
                facts_with_relevant_rel = [(weight, fact) for weight, fact in observed_weights_and_facts if fact['rel'] == relevant_rel]
                observation_vectors = [weight_and_index_to_observation_vector(self.num_object_entities, weight, self.obj_to_index[fact['e2']]) for weight, fact in facts_with_rel]
                observation_vector = np.stack(observation_vectors).sum(axis=0)
                relevant_properties = [fact for _, fact in facts_with_relevant_rel]
                relevant_objects = np.array([fact['e2'] for fact in relevant_properties])
                for relevant_object in set(relevant_objects):
                    property_proportion = np.mean(relevant_objects==relevant_object) # THIS IS THE OLD PROPERTY FREQUENCY
                    distr = self.o_given_r_property[rel][relevant_object]
                    reweighted_vector = observation_vector * property_proportion
                    distr.observe_vector(reweighted_vector)
    
    def increment_o_given_r_property_distributions(self, list_of_weights_and_facts):
        '''
        This function is used when new evidence is added to the model. We remove the influence of the old {s,r1,*} observations and add the new {s,r1,*} observations
        '''
        # find the new subjects that we need to update, along with what relations we've seen for these facts
        new_facts_grouped_by_subject = self.group_facts_by_subject_entity(list_of_weights_and_facts)
        new_subject_entity_to_seen_relations = self.identify_relations_per_subject([fact for weight, fact in list_of_weights_and_facts])
        # loop through and REcompute the evidence provided by the set of facts for each subject that we want to update
        for subj in new_facts_grouped_by_subject.keys():
            seen_rels = list(self.subject_entity_to_seen_relations[subj]) + list(new_subject_entity_to_seen_relations[subj])
            observed_related_rel_pairs = [(seen_rel, self.relevant_relations_dict[seen_rel]) for seen_rel in seen_rels if self.relevant_relations_dict[seen_rel] in seen_rels]
            all_observed_weights_and_facts = self.facts_grouped_by_subject[subj] + new_facts_grouped_by_subject[subj]
            old_observed_weights_and_facts = self.facts_grouped_by_subject[subj]
            # remove the old observations from the distributions
            for related_rel_pair in observed_related_rel_pairs:
                rel, relevant_rel = related_rel_pair
                facts_with_rel = [(weight, fact) for weight, fact in old_observed_weights_and_facts if fact['rel'] == rel]
                facts_with_relevant_rel = [(weight, fact) for weight, fact in old_observed_weights_and_facts if fact['rel'] == relevant_rel]
                observation_vectors = [weight_and_index_to_observation_vector(self.num_object_entities, weight, self.obj_to_index[fact['e2']]) for weight, fact in facts_with_rel]
                observation_vector = np.stack(observation_vectors).sum(axis=0)
                relevant_properties = [fact for _, fact in facts_with_relevant_rel]
                relevant_objects = np.array([fact['e2'] for fact in relevant_properties])
                for relevant_object in set(relevant_objects):
                    property_proportion = np.mean(relevant_objects==relevant_object) # THIS IS THE OLD PROPERTY FREQUENCY
                    distr = self.o_given_r_property[rel][relevant_object]
                    reweighted_vector = observation_vector * property_proportion
                    distr.observe_vector(-reweighted_vector) # observing the NEGATIVE of the observation vector removes the observation from the distribution
            # add the new COMBINED observations to the distributions
            for related_rel_pair in observed_related_rel_pairs:
                rel, relevant_rel = related_rel_pair
                facts_with_rel = [(weight, fact) for weight, fact in all_observed_weights_and_facts if fact['rel'] == rel]
                facts_with_relevant_rel = [(weight, fact) for weight, fact in all_observed_weights_and_facts if fact['rel'] == relevant_rel]
                observation_vectors = [weight_and_index_to_observation_vector(self.num_object_entities, weight, self.obj_to_index[fact['e2']]) for weight, fact in facts_with_rel]
                observation_vector = np.stack(observation_vectors).sum(axis=0)
                relevant_properties = [fact for _, fact in facts_with_relevant_rel]
                relevant_objects = np.array([fact['e2'] for fact in relevant_properties])
                for relevant_object in set(relevant_objects):
                    property_proportion = np.mean(relevant_objects==relevant_object) # THIS IS THE OLD PROPERTY FREQUENCY
                    distr = self.o_given_r_property[rel][relevant_object]
                    reweighted_vector = observation_vector * property_proportion
                    distr.observe_vector(reweighted_vector)

    def decrement_o_given_r_property_distributions(self, list_of_weights_and_facts):
        '''
        This function is used when specific evidence is REMOVED from the model. We remove the influence of the COMBINED {s,r1,*} observations and add back the OLD {s,r1,*} observations
        '''
        # find the new subjects that we need to update, along with what relations we've seen for these facts
        new_facts_grouped_by_subject = self.group_facts_by_subject_entity(list_of_weights_and_facts)
        new_subject_entity_to_seen_relations = self.identify_relations_per_subject([fact for weight, fact in list_of_weights_and_facts])
        # loop through and REcompute the evidence provided by the set of facts for each subject that we want to update
        for subj in new_facts_grouped_by_subject.keys():
            seen_rels = list(self.subject_entity_to_seen_relations[subj]) + list(new_subject_entity_to_seen_relations[subj])
            observed_related_rel_pairs = [(seen_rel, self.relevant_relations_dict[seen_rel]) for seen_rel in seen_rels if self.relevant_relations_dict[seen_rel] in seen_rels]
            all_observed_weights_and_facts = self.facts_grouped_by_subject[subj] + new_facts_grouped_by_subject[subj]
            old_observed_weights_and_facts = self.facts_grouped_by_subject[subj]
            # remove the COMBINED observations from the distributions
            for related_rel_pair in observed_related_rel_pairs:
                rel, relevant_rel = related_rel_pair
                facts_with_rel = [(weight, fact) for weight, fact in all_observed_weights_and_facts if fact['rel'] == rel]
                facts_with_relevant_rel = [(weight, fact) for weight, fact in all_observed_weights_and_facts if fact['rel'] == relevant_rel]
                observation_vectors = [weight_and_index_to_observation_vector(self.num_object_entities, weight, self.obj_to_index[fact['e2']]) for weight, fact in facts_with_rel]
                observation_vector = np.stack(observation_vectors).sum(axis=0)
                relevant_properties = [fact for _, fact in facts_with_relevant_rel]
                relevant_objects = np.array([fact['e2'] for fact in relevant_properties])
                for relevant_object in set(relevant_objects):
                    property_proportion = np.mean(relevant_objects==relevant_object) # THIS IS THE OLD PROPERTY FREQUENCY
                    distr = self.o_given_r_property[rel][relevant_object]
                    reweighted_vector = observation_vector * property_proportion
                    distr.observe_vector(-reweighted_vector) # observing the NEGATIVE of the observation vector removes the observation from the distribution
            # add back the OLD observations to the distributions
            for related_rel_pair in observed_related_rel_pairs:
                rel, relevant_rel = related_rel_pair
                facts_with_rel = [(weight, fact) for weight, fact in old_observed_weights_and_facts if fact['rel'] == rel]
                facts_with_relevant_rel = [(weight, fact) for weight, fact in old_observed_weights_and_facts if fact['rel'] == relevant_rel]
                observation_vectors = [weight_and_index_to_observation_vector(self.num_object_entities, weight, self.obj_to_index[fact['e2']]) for weight, fact in facts_with_rel]
                observation_vector = np.stack(observation_vectors).sum(axis=0)
                relevant_properties = [fact for _, fact in facts_with_relevant_rel]
                relevant_objects = np.array([fact['e2'] for fact in relevant_properties])
                for relevant_object in set(relevant_objects):
                    property_proportion = np.mean(relevant_objects==relevant_object) # THIS IS THE OLD PROPERTY FREQUENCY
                    distr = self.o_given_r_property[rel][relevant_object]
                    reweighted_vector = observation_vector * property_proportion
                    distr.observe_vector(reweighted_vector)

    def check_conditional_distribution_modes(self, only_relevant_rel=None, highlight_obj=''):
        # check distribution modes for conditional distributions
        for rel, relevant_rel in self.relevant_relations_dict.items():
            if only_relevant_rel is not None:
                if relevant_rel != only_relevant_rel:
                    continue
            print(f"RELs: {rel} | {relevant_rel}")
            for obj_no, relevant_obj in enumerate(self.object_entities):
                if rel not in self.o_given_r_property:
                    continue
                distr = self.o_given_r_property[rel][relevant_obj]
                n = distr.get_sample_size()
                if n > 0:
                    max_idx = distr.get_posterior_argmax()
                    if relevant_obj == highlight_obj:
                        print(f"  ***   relevant obj: {relevant_obj} | n = {n} | mode: {self.object_entities[max_idx]}")
                    else:
                        print(f"relevant obj: {relevant_obj} | n = {n} | mode: {self.object_entities[max_idx]}")
                    # if rel == 'occupation' and relevant_obj == 'turkiye':
                    #     breakpoint()

    def get_requested_obj_with_downstream_fact_change(self, upstream_obj, upstream_rel, downstream_rel):
        # check distribution modes for conditional distributions
        assert self.relevant_relations_dict[downstream_rel] == upstream_rel, "Providing incorrect upstream/downstream rel pair"
        orig_GT_downstream_obj = self.o_given_r_property[downstream_rel][upstream_obj]
        eligigible_objects = []
        for relevant_obj in self.object_entities:
            distr = self.o_given_r_property[downstream_rel][relevant_obj]
            n = distr.get_sample_size()
            if n > 0:
                mode_idx = distr.get_posterior_argmax()
                new_modal_obj = self.object_entities[mode_idx]
                if new_modal_obj != orig_GT_downstream_obj:
                    eligigible_objects.append(relevant_obj)
        return self.rng.choice(eligigible_objects)

def weight_and_index_to_observation_vector(num_object_entities, weight, index):
    # a weight of 1 means that we count one observation in favor of the object indicated by the index
    # a weight of -1 means we consider to have observed one datapoint implying that the object is not the one indicated by the index. this means we spread out one observation across all other objects
    # a weight of 0 means that the observation counts as nothing
    # a weight of 'REMOVE' means to use a negative observation vector in order to undo the effect of observing a fact
    # if we have a fact with a 2/3 weight, that means we update 2/3 toward the statement s r o, and 1/3 toward the statement not s r o, which is equal to s r o with weight -1/3
    try:
        assert weight == -1 or (weight > 0 and weight < 1) or (weight >= 1)
    except:
        breakpoint()
        assert utils.smalldiff(weight, -1) or (weight > 0 and weight < 1) or weight >= 1
    weight_fractional = (weight > 0 and weight < 1)
    if weight >= 1:
        one_hot = np.zeros(num_object_entities)
        one_hot[index] = weight
        weighted_observation_vector = one_hot
    elif weight == -1:
        all_but_index = np.ones(num_object_entities)
        all_but_index[index] = 0
        multiplier = 1 / (num_object_entities-1)
        all_but_index = multiplier * all_but_index
        assert np.abs(all_but_index.sum() - 1) < .0001
        weighted_observation_vector = all_but_index
    elif weight_fractional:
        positive_component = weight
        negative_component = 1 - weight
        # make the "positive" weighted observation in favor of the observed object index
        one_hot = np.zeros(num_object_entities)
        one_hot[index] = 1
        positive_weighted_observation_vector = positive_component * one_hot
        # make the "negative" weighted observation in favor of other classes
        all_but_index = np.ones(num_object_entities)
        all_but_index[index] = 0
        multiplier = 1 / (num_object_entities-1)
        all_but_index = multiplier * all_but_index
        # assert all_but_index.sum() == 1
        assert np.abs(all_but_index.sum() - 1) < .0001
        negative_weighted_observation_vector = negative_component * all_but_index
        weighted_observation_vector = positive_weighted_observation_vector + negative_weighted_observation_vector
        # assert weighted_observation_vector.sum() == 1
        assert np.abs(weighted_observation_vector.sum() - 1) < .0001
    return weighted_observation_vector
