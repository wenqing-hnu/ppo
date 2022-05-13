import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import pandas as pd
import os
import argparse
import json

def plot_data(data_set, y_value=None, sub_axis=False, save_name=None):
    save_dir = './pictures/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    all_data = pd.concat(data_set, axis=0, join='inner', ignore_index=True)
    show_label = False
    labels = None
    # labels = ['PPO-$\lambda$-MPC', 'PPO-$\lambda$(baseline)', 'SACD-$\lambda$-TM', 'SACD-$\lambda$(baseline)',
    #           'SACD-$\lambda$-MPC']
    labels = ['$\sigma=1$', '$\sigma=5$', '$\sigma=10$']
    label_size = 28
    ticks_size = 24
    legend_size = 22
    line_size = 2.5
    # for crash ratio
    ylimit = (-0.005, 0.5)
    ylabel = 'Crash Ratio'
    fig = plt.figure()
    k = 1
    for i in y_value:
        if i == 'AverageEpCost':
            ylimit = (-0.005, 0.8)
            ylabel = 'Average Cost'
        elif i == 'AverageEpRet':
            ylimit = (-2, 10)
            ylabel = 'Average Reward'

        sns.set(style="white")
        sns.set_context(rc={"lines.linewidth": line_size})
        hue_order = None
        title = None
        fig.set_size_inches(10, 6)
        if save_name == 'ppo_sigma':
            hue_order = ['PPO_mpc_0th', 'PPO_mpc', 'PPO_mpc_10th']
        elif save_name == 'sacd_sigma':
            hue_order = ['SAC_SACD-TDn-MPC-0th', 'SAC_nsteps_mpc', 'SAC_SACD-TDn-MPC-10th']
        # fig.add_subplot(1, 3, k)
        ax = sns.lineplot(data=all_data, x='Epoch', y=i, hue='Exp_name', hue_order=hue_order, legend=False)
        ax.grid(True)
        plt.xlabel('TotalEnvInteracts(Million)', fontsize=label_size)
        plt.ylabel(ylabel, fontsize=label_size)
        plt.xticks(np.linspace(1, 125, 6), ['0', '0.1', '0.2', '0.3', '0.4', '0.5'], fontsize=ticks_size)
        plt.yticks(fontsize=ticks_size)
        plt.ylim(ylimit)
        labels = ['Roundabout 1']
        # if not show_label:
        #     plt.legend(labels=labels, fontsize=legend_size, title=title, title_fontsize=24)
        #     show_label = True

        # add subaxis
        if sub_axis:
            if i == 'Crash_ratio':
                axins = inset_axes(ax, width="45%", height="35%", loc='lower left',
                                   bbox_to_anchor=(2.5 / 5, 1.5 / 5, 0.8, 0.7),
                                   bbox_transform=ax.transAxes)

                subax = sns.lineplot(data=all_data, x='Epoch', y='Crash_ratio', hue='Exp_name', ax=axins,
                                     legend=False)
                subax.axes.yaxis.set_visible(False)
                subax.axes.xaxis.set_visible(False)

                subax.set_xlim(100, 125)
                subax.set_ylim(0, 0.04)

                mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec='k', lw=1)

            elif i == 'AverageEpCost':
                axins = inset_axes(ax, width="45%", height="35%", loc='lower left',
                                   bbox_to_anchor=(3 / 5, 4.5 / 8, 0.8, 0.7),
                                   bbox_transform=ax.transAxes)
                subax = sns.lineplot(data=all_data, x='Epoch', y='AverageEpCost', hue='Exp_name', ax=axins,
                                     legend=False)
                subax.axes.yaxis.set_visible(False)
                subax.axes.xaxis.set_visible(False)

                subax.set_xlim(100, 125)
                subax.set_ylim(0, 0.06)

                mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec='k', lw=1)

            elif i == 'AverageEpRet':
                axins = inset_axes(ax, width="45%", height="35%", loc='lower left',
                                   bbox_to_anchor=(3 / 5, 9 / 17, 0.8, 0.7),
                                   bbox_transform=ax.transAxes)
                subax = sns.lineplot(data=all_data, x='Epoch', y='AverageEpRet', hue='Exp_name', ax=axins,
                                     legend=False)
                subax.axes.yaxis.set_visible(False)
                subax.axes.xaxis.set_visible(False)

                subax.set_xlim(100, 125)
                subax.set_ylim(14.5, 15.8)

                mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec='k', lw=1)

        k += 1

    plt.savefig(os.path.join(save_dir, save_name + 'yzy'), dpi=600, format='pdf', bbox_inches='tight')

    plt.show()

def get_data(args):
    all_data = []
    for i in args.algo:
        path = args.file_path + i
        for root, dirs, files in os.walk(path):
            if 'progress.txt' in files:
                data_path = os.path.join(root, 'progress.txt')
                exp_data = pd.read_table(data_path)
                # make sure epoch starts from 1
                if exp_data['Epoch'][0] == 0:
                    exp_data['Epoch'] += 1

                exp_name = 'NAN'

                try:
                    config_path = open(os.path.join(root, 'config.json'))
                    config = json.load(config_path)
                    exp_name = config['exp_name']
                except:
                    print('No file named config.json')

                exp_data.insert(len(exp_data.columns), 'Exp_name', exp_name)
                all_data.append(exp_data)

    return all_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default='data/')
    parser.add_argument("--algo", type=list, default=['Roundabout_ppo_exs'])
    parser.add_argument("--y_value", type=list, default=['AverageEpRet'])
    parser.add_argument('--add_subaxis', type=bool, default=False)
    parser.add_argument("--save_name", type=str, default='roundabout_exs')
    args = parser.parse_args()

    data_set = get_data(args)
    # all_data = pd.concat(data_set, axis=0, join='inner', ignore_index=True)
    plot_data(data_set,
              y_value=args.y_value,
              sub_axis=args.add_subaxis,
              save_name=args.save_name)
