class KnowledgeGraph:
    '''
    Used for storing entities and their properties. Also computes/stores basic attributes of the knowledge graph
    '''
    def __init__(self, entity_info_dict=dict(), list_of_facts=None, entity_dict=None, relation_dict=None):
        self.num_facts = 0
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.all_entities = set()
        self.all_subject_entities = set()
        self.all_relations = set()
        self.relations_list = list() # keeps ordering before/after translation, and for use with computing co-occurence statistics
        self.all_object_entities = set()
        self.entity_info_dict = dict()
        self.add_entity_info_dict(entity_info_dict)
        if list_of_facts is not None:
            self.add_list_of_facts(list_of_facts)
        assert self.num_facts == self.compute_num_facts()

    def add_entity_info_dict(self, entity_info_dict):
        subject_entities = list(entity_info_dict.keys())
        for e1 in subject_entities:
            info_dict = entity_info_dict.pop(e1)
            facts = [{'e1': e1, 'rel': rel, 'e2': e2} for (rel, e2) in info_dict.items()]
            for fact in facts:
                self.add_fact(fact)
        del subject_entities

    def compute_num_facts(self):
        # compute number of available 1:1 (e,r,o) facts available
        num_facts = 0
        for e1, info_dict in self.entity_info_dict.items():
            num_facts += len(info_dict)
        return num_facts
    
    def add_fact(self, fact):
        e1 = fact['e1']
        rel = fact['rel']
        e2 = fact['e2']
        if e1 in self.entity_info_dict:
            if rel not in self.entity_info_dict[e1]:
                self.entity_info_dict[e1][rel] = e2
                self.num_facts += 1
            else:
                assert self.entity_info_dict[e1][rel] == e2, f"Trying to add a fact that conflicts with known fact. New fact: {fact} vs. old knowledge: {self.entity_info_dict[e1]}"
        else:
            self.entity_info_dict[e1] = {rel: e2}
            self.num_facts += 1
        self.all_entities.add(e1)
        self.all_entities.add(e2)
        self.all_subject_entities.add(e1)
        self.all_object_entities.add(e2)
        self.all_relations.add(rel)
        if rel not in self.relations_list:
            self.relations_list.append(rel)

    def add_list_of_facts(self, facts):
        for fact in facts:
            self.add_fact(fact)

    def unroll_atomic_facts(self):
        atomic_facts = []
        for e1, info_dict in self.entity_info_dict.items():
            for rel, e2 in info_dict.items():
                atomic_facts.append({
                    'e1': e1,
                    'rel': rel,
                    'e2': e2,
                })
        return atomic_facts
    
    def set_alias_dicts(self, entity_dict, relation_dict):
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
    
    def translate_into_english(self):
        assert self.entity_dict is not None
        assert self.relation_dict is not None
        original_subjects = list(self.entity_info_dict.keys())
        # translate entities/rels
        self.all_entities = set([self.entity_dict[e][0] for e in self.all_entities])
        self.all_subject_entities = set([self.entity_dict[e1][0] for e1 in self.all_subject_entities])
        self.all_object_entities = set([self.entity_dict[e2][0] for e2 in self.all_object_entities])
        self.all_relations = set([self.relation_dict[rel][0] for rel in self.all_relations])
        self.relations_list = [self.relation_dict[rel][0] for rel in self.relations_list]
        # translate self.entity_info_dict
        for e1 in original_subjects:
            info_dict = self.entity_info_dict.pop(e1)
            english_facts = []
            for rel, e2 in info_dict.items():
                english_fact = {
                    'e1': self.entity_dict[e1][0],
                    'rel': self.relation_dict[rel][0],
                    'e2': self.entity_dict[e2][0],
                }
                english_facts.append(english_fact)
            for fact in english_facts:
                self.add_fact(fact)
        del original_subjects

    def cap_object_entity_token_len(self, tokenizer, max_tokens):
        for e1, info_dict in self.entity_info_dict.items():
            for rel, e2 in info_dict.items():
                e2_tokens = tokenizer.encode(e2, add_special_tokens=False)
                if len(e2_tokens) > max_tokens:
                    if e2 in self.all_entities:
                        self.all_entities.remove(e2)
                    if e2 in self.all_object_entities:
                        self.all_object_entities.remove(e2)
                    e2_tokens = e2_tokens[:max_tokens]
                    e2 = tokenizer.decode(e2_tokens)
                    self.all_entities.add(e2)
                    self.all_object_entities.add(e2)
                self.entity_info_dict[e1][rel] = e2
        