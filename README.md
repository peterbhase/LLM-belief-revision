This is the codebase for the paper [Fundamental Problems With Model Editing: How Should Rational Belief Revision Work in LLMs?]().

#### Install Requirements

```
pip install -r requirements.txt
```

### Experiments

Pretraining and editing experiments save results to `training_logs` and `outputs` directories.

#### Download Wikidata5m

First, set the environment variable `DATA_DIR`. Then, run

```
scripts/download_wikidata.sh $DATA_DIR
```

#### Fit a model to 10k facts, with no logically complex sentences, without model editing evaluation.

Note that you must supply the `--model_dir`, `--data_dir`, and `--cache_dir` args for saving/storing models by setting the respective environment variables.

```
python main.py --num_atomic_facts 10000 \ 
    --num_relations 10 \
    --n_base_fact_samples 10 \
    -mgtp .6 \
    --model mistral-83m \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    -gaf 1 \
    --max_sentences_per_doc 1 \
    --add_is_true_false_sentences false \
    --add_not_sentences false \
    --add_and_sentences false \
    --add_or_sentences false \
    --lr 5e-5 \
    -tb 5e7 \
    --read_n_facts 2e6 \
    --write_corpus true \
    --do_model_editing false \
    --do_train true \
    --update_lr 1e-4 \
    --model_dir $MODEL_DIR \
    --data_dir $DATA_DIR \
    --cache_dir $CACHE_DIR \
    --debug True
```

#### Fit a model to 100k facts, with logically complex sentences, with model editing evaluation. This is the experiment setting in the paper.

```
python main.py --num_atomic_facts 100000 \ 
    --num_relations 10 \
    --eval_size 5000 \
    --n_base_fact_samples 10 \
    -mgtp .6 \
    --model mistral-83m \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    -gaf 2 \
    --max_sentences_per_doc 10 \
    -ncs 20 \
    --k_homogenous_doc_order_resamples 5 \
    --add_is_true_false_sentences true \
    --add_not_sentences true \
    --add_and_sentences true \
    --add_or_sentences true \
    --lr 5e-5 \
    -tb 1e9 \
    --read_n_facts 2e6 \
    --write_corpus true \
    --do_model_editing true \
    --do_train true \
    --model_dir $MODEL_DIR \
    --data_dir $DATA_DIR \
    --cache_dir $CACHE_DIR \
    --update_lr 1e-4
```

### Data Analysis

We provide the R markdown file used for data analysis. The `analysis.Rmd` is used to recreate the main editing results table, qualitative examples of editing performance, and pretraining performance plots in the paper.














