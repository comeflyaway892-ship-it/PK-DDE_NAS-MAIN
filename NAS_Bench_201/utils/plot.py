import os
import numpy as np
import matplotlib.pyplot as plt

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_denoise(save_dir, avg_acc_trace, max_acc_trace, valid_rate_trace, uniq_rate_trace, seed, dataset):
    create_dir(save_dir)
    title_fontsize = 26
    label_fontsize = 26
    tick_fontsize = 26
    legend_fontsize = 26
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # 绘制平均适应度值变化图
    ax1.plot(np.array(avg_acc_trace), color='blue', label='avg_acc')
    ax1.set_xlabel('Iteration', fontsize=label_fontsize)
    ax1.set_ylabel('Accuracy', fontsize=label_fontsize)
    ax1.tick_params(labelsize=tick_fontsize)
    ax1.legend(fontsize=legend_fontsize)
    ax1.set_title('Average Accuracy Trace for Denoise', fontsize=title_fontsize)

    # 绘制最大适应度值变化图 
    ax2.plot(np.array(max_acc_trace), color='red', label='max_acc')
    ax2.set_xlabel('Iteration', fontsize=label_fontsize)
    ax2.set_ylabel('Accuracy', fontsize=label_fontsize)
    ax2.tick_params(labelsize=tick_fontsize)
    ax2.legend(fontsize=legend_fontsize)
    ax2.set_title('Max Accuracy Trace for Denoise', fontsize=title_fontsize)

    # 绘制有效率变化图 
    ax3.plot(np.array(valid_rate_trace), color='orange', label='valid_rate')
    ax3.set_xlabel('Denoise Iteration', fontsize=label_fontsize)
    ax3.set_ylabel('Valid Rate', fontsize=label_fontsize)
    ax3.tick_params(labelsize=tick_fontsize)
    ax3.legend(fontsize=legend_fontsize)
    ax3.set_title('Valid Rate Trace', fontsize=title_fontsize)

    # 绘制unique率变化图
    ax4.plot(np.array(uniq_rate_trace), color='green', label='uniq_rate')
    ax4.set_xlabel('Denoise Iteration', fontsize=label_fontsize)
    ax4.set_ylabel('Unique Rate', fontsize=label_fontsize)
    ax4.tick_params(labelsize=tick_fontsize)
    ax4.legend(fontsize=legend_fontsize)
    ax4.set_title('Unique Rate Trace', fontsize=title_fontsize)

    # 保存图片
    plt.savefig(f'{save_dir}/{dataset}_seed_{seed}.png', bbox_inches='tight')
    plt.close()
