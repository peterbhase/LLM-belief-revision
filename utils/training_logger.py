import pandas as pd
import matplotlib.pyplot as plt
import utils
import numpy as np

class TrainingLogger():
    """
    Report stores evaluation results during the training process as text files.
    """

    def __init__(self, args, file_path, experiment_name, overwrite_existing=True):
        self.fn = file_path
        self.args = args
        self.experiment_name = experiment_name
        self.max_len = 11
        self.old_running_time = 0
        self.curr_speed = 0
        self.columns = ["epoch", "n_tokens", "n_sentences", "step", "train_loss", "eval_loss", "o_given_s_r_loss", "eval_acc", "LR", "seq_loss",
                        "o_given_s_r_acc", 'A_TF_case_acc', 'A_and_B_case_acc', 'A_or_B_case_acc', 'B_TF_case_acc', 'B_and_A_case_acc', "not_A_case_acc",
                        'generative_prob_error',
                        'data_frequency_o_given_s_r_error',
                        'posterior_prob_o_given_s_r_error',
                        'data_frequency_marginalized_error',
                        'posterior_prob_marginalized_error',
                        'data_frequency_true_conditional_error',
                        'posterior_prob_true_conditional_error',
                        'commutation_loss', 'disjunction_loss', 'multiplication_loss', 'negation_loss', 'TF_coherence_loss']
        self.log_df = pd.DataFrame(columns=self.columns)
        self.subset_to_log_df = {}

    def add_to_log(self, stats):
        assert sorted(list(stats.keys())) == sorted(self.columns), f"please add missing columns to TrainingLogger.columns or stats. Compare values in self.columns vs stats.keys(): {' --- '.join([f'  {x} vs. {y}' for x,y in zip(sorted(self.columns), sorted(stats.keys()))])}"
        stats_df = pd.DataFrame([stats])
        self.log_df = pd.concat([self.log_df, stats_df], ignore_index=True)
        self.save_log()

    def add_subset_stats_to_logs(self, subset_stats):
        for k,v in subset_stats.items():
            v_df = v
            if k not in self.subset_to_log_df:
                self.subset_to_log_df[k] = v_df
            else:
                self.subset_to_log_df[k] = pd.concat([self.subset_to_log_df[k], v_df], ignore_index=True)

    def get_last_eval_result(self):
        return self.log_df.iloc[-1]

    def save_log(self):
        self.log_df.to_csv(self.fn, index=False)

    def print_training_prog(self, train_stats, epoch, num_epochs, batch_num, 
        n_batches, running_time, est_epoch_run_time, forward_time, current_LR=None, gpu_mem=None):
        last_batch = batch_num == n_batches-1
        print_str = f" Epoch {epoch}/{num_epochs} | Batch: {batch_num+1}/{n_batches}"
        for k, v in train_stats.items():
            # if 'acc' in k or k == 'train_loss':
            if k == 'loss' or 'acc' in k:
                print_str += f" | {k.capitalize()}: {v:.2f}"
        if current_LR is not None:
            print_str += f" | LR: {current_LR:.6f} | Runtime: {running_time/60:.1f} min. / {est_epoch_run_time/60:.1f} min. | Forward time: {forward_time:.5f} sec."
        else:
            print_str += f" | Runtime: {running_time/60:.1f} min. / {est_epoch_run_time/60:.1f} min. | Forward time: {forward_time:.5f} sec."
        if gpu_mem:
            print_str += f" | Mem: {gpu_mem}"
        print(print_str, end='\r' if not last_batch else '\n')
        
    def print_epoch_scores(self, epoch, scores):
        print(f"Epoch: {epoch}")
        print("-" * 50)
        num_scores = len(scores)
        max_name_width = 14 # max(len(name) for name in scores.keys())
        num_columns = min(num_scores, 4)
        num_rows = (num_scores + num_columns - 1) // num_columns
        score_names = list(scores.keys())
        score_values = list(scores.values())
        for row in range(num_rows):
            for col in range(num_columns):
                index = row + col * num_rows
                if index < num_scores:
                    name = score_names[index]
                    value = score_values[index]
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                    print(f"{name[:max_name_width]:<{max_name_width}} : {formatted_value:>{6}}", end="  ")
            print()
        print("-" * 50)

    def save_plots(self, plot_all_subsets=False):
        # save plots of train_loss/eval_loss/eval_acc vs. steps/n_tokens/n_sentences/epochs
        main_outcomes = ['train_loss', 'eval_loss', 'eval_acc', 'o_given_s_r_loss']
        complex_sent_outcomes = [
            'o_given_s_r_acc',
            'A_and_B_case_acc',
            'B_and_A_case_acc',
            'A_or_B_case_acc',
            'not_A_case_acc',
            'A_TF_case_acc',
            'B_TF_case_acc',
        ]
        logical_coherence = [
            'eval_loss',
            'multiplication_loss',
            'commutation_loss',
            'disjunction_loss',
            'negation_loss',
            'TF_coherence_loss',
        ]
        probabilistic_coherence = [
            'eval_loss',
            'generative_prob_error',
            'data_frequency_o_given_s_r_error',
            'posterior_prob_o_given_s_r_error',
            'data_frequency_marginalized_error',
            'posterior_prob_marginalized_error',
            'data_frequency_true_conditional_error',
            'posterior_prob_true_conditional_error',
        ]
        plot_against_epoch = not (self.args.num_evals > self.args.num_train_epochs)
        plot_against_tokens = not plot_against_epoch or self.args.training_budget_in_tokens >= 5e7
        x_vars = []
        if plot_against_epoch:  x_vars.append('epoch')
        if plot_against_tokens: x_vars.append('n_tokens')

        # decide which data subsets to plot
        if plot_all_subsets:
            self.subset_to_log_df['all'] = self.log_df
            names_and_dfs = [(k,v) for k,v in self.subset_to_log_df.items() if 'epoch' in v.columns]
        else:
            names_and_dfs = [('all', self.log_df)]

        for name, log_df in names_and_dfs:
            if name == 'seen':
                continue

            # overlay the eval_acc, train_loss, and eval_loss variables -- only do this for 'all' subset
            if name == 'all':
                for x_var in x_vars:
                    fig, ax1 = plt.subplots()
                    ax2 = ax1.twinx()
                    plot_name = f"plt_{self.experiment_name}_main-outcomes_vs_{x_var}"
                    for outcome in main_outcomes:
                        if 'loss' in outcome:
                            # loss_divide_factor = 1 if outcome == 'o_given_s_r_loss' else 10
                            ax1.plot(log_df[x_var], log_df[outcome], label=outcome)
                        else:
                            ax2.plot(log_df[x_var], log_df[outcome], label=outcome, color='red')
                    ax1.set_xlabel(x_var)
                    ax1.set_ylabel("Loss", rotation=0)
                    max_loss = 10 if log_df['eval_loss'].min() > 5 else 5
                    ax1.set_ylim(0, max_loss)
                    ax2.set_xlabel(x_var)
                    ax2.set_ylabel("Acc", rotation=0)
                    ax2.set_ylim(0, 1.04)
                    lines, labels = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax2.legend(lines + lines2, labels + labels2, loc='lower left')
                    fig.suptitle(f"Model Performance vs. {x_var}")

                    # find the peak value of the curve
                    peak_value = log_df['eval_acc'].max()
                    peak_index = log_df['eval_acc'].idxmax()
                    peak_x_val = log_df[x_var].iloc[peak_index]
                    # annotate the peak value on the plot
                    ax2.text(.5, 1.03, f'acc: {peak_value:.2f} (at x={int(peak_x_val)})',
                        transform=ax2.transAxes, horizontalalignment='center')
                    ax2.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
                    ax2.axvline(x=peak_x_val, color='red', linestyle='--', linewidth=0.5)

                    # save the plot to a PDF file
                    filepath = f'training_logs/{plot_name}'
                    # plt.savefig(filepath+'.pdf', format='pdf')
                    plt.savefig(filepath+'.png', format='png')
                    plt.clf()
                    ax1.cla()
                    ax2.cla()
                    plt.close()

            # plot complex sentence outcomes
            have_outcomes = [outcome for outcome in complex_sent_outcomes if outcome in log_df.columns]
            if len(have_outcomes) > 0:
                for x_var in x_vars:
                    plot_name = f"plt_{self.experiment_name}_{name}_complex-outcomes_vs_{x_var}"
                    fig, ax = plt.subplots()
                    for outcome in have_outcomes:
                        ax.plot(log_df[x_var], log_df[outcome], label=outcome)
                    ax.set_xlabel(x_var)
                    ax.set_ylabel("Acc")
                    fig.suptitle(f"Acc outcomes vs {x_var}")
                    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                    ax.set_ylim(0, 1.1)
                    # save the plot to a PDF file
                    filepath = f'training_logs/{plot_name}.png'
                    plt.savefig(filepath, format='png', bbox_inches='tight', pad_inches=1)
                    plt.clf()
                    ax.cla()
                    plt.close()

            # plot probabilistic coherence outcomes
            have_outcomes = [outcome for outcome in probabilistic_coherence if outcome in log_df.columns]
            if len(have_outcomes) > 0:
                for x_var in x_vars:
                    plot_name = f"plt_{self.experiment_name}_{name}_prob-outcomes_vs_{x_var}"
                    fig, ax = plt.subplots()
                    for outcome in have_outcomes:
                        # if outcome in log_df.columns:
                        ax.plot(log_df[x_var], log_df[outcome], label=outcome)
                    ax.set_xlabel(x_var)
                    ax.set_ylabel("Loss")
                    fig.suptitle(f"Loss vs {x_var}")
                    ax.set_ylim(0, 1)
                    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                    # save the plot to a PDF file
                    filepath = f'training_logs/{plot_name}.png'
                    plt.savefig(filepath, format='png', bbox_inches='tight', pad_inches=1)
                    plt.clf()
                    ax.cla()
                    plt.close()

            # plot logical coherence outcomes
            have_outcomes = [outcome for outcome in logical_coherence if outcome in log_df.columns]
            if len(have_outcomes) > 0:
                for x_var in x_vars:
                    plot_name = f"plt_{self.experiment_name}_{name}_logic-outcomes_vs_{x_var}"
                    fig, ax = plt.subplots()
                    for outcome in filter(lambda outcome: outcome in log_df.columns, logical_coherence):
                        ax.plot(log_df[x_var], log_df[outcome], label=outcome)
                    ax.set_xlabel(x_var)
                    ax.set_ylabel("Loss")
                    fig.suptitle(f"Loss vs {x_var}")
                    ax.set_ylim(0, 1.1)
                    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                    # save the plot to a PDF file
                    filepath = f'training_logs/{plot_name}.png'
                    plt.savefig(filepath, format='png', bbox_inches='tight', pad_inches=1)
                    plt.clf()
                    ax.cla()
                    plt.close()