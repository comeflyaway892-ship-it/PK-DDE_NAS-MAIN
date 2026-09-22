import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


def setup_chinese_font():
    font_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            return

    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "SimHei",
        "Microsoft YaHei",
        "WenQuanYi Micro Hei",
        "Droid Sans Fallback",
        "DejaVu Sans",
    ]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_denoise(
    save_dir,
    avg_acc_trace,
    max_acc_trace,
    valid_rate_trace,
    uniq_rate_trace,
    seed,
    dataset,
):
    setup_chinese_font()
    create_dir(save_dir)
    title_fontsize = 26
    label_fontsize = 26
    tick_fontsize = 26
    legend_fontsize = 26
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # 绘制平均准确率变化图
    ax1.plot(np.array(avg_acc_trace), color="blue", label="平均准确率")
    ax1.set_xlabel("迭代次数", fontsize=label_fontsize)
    ax1.set_ylabel("准确率", fontsize=label_fontsize)
    ax1.tick_params(labelsize=tick_fontsize)
    ax1.legend(fontsize=legend_fontsize)
    ax1.set_title("去噪过程中新生成架构的平均准确率变化", fontsize=title_fontsize)

    # 绘制最大准确率变化图
    ax2.plot(np.array(max_acc_trace), color="red", label="最大准确率")
    ax2.set_xlabel("迭代次数", fontsize=label_fontsize)
    ax2.set_ylabel("准确率", fontsize=label_fontsize)
    ax2.tick_params(labelsize=tick_fontsize)
    ax2.legend(fontsize=legend_fontsize)
    ax2.set_title("去噪过程中新生成架构的最大准确率变化", fontsize=title_fontsize)

    # 绘制有效率变化图
    ax3.plot(np.array(valid_rate_trace), color="orange", label="有效率")
    ax3.set_xlabel("迭代次数", fontsize=label_fontsize)
    ax3.set_ylabel("有效率", fontsize=label_fontsize)
    ax3.tick_params(labelsize=tick_fontsize)
    ax3.legend(fontsize=legend_fontsize)
    ax3.set_title("去噪过程中新生成架构的有效率变化", fontsize=title_fontsize)

    # 绘制唯一率变化图
    ax4.plot(np.array(uniq_rate_trace), color="green", label="唯一率")
    ax4.set_xlabel("迭代次数", fontsize=label_fontsize)
    ax4.set_ylabel("唯一率", fontsize=label_fontsize)
    ax4.tick_params(labelsize=tick_fontsize)
    ax4.legend(fontsize=legend_fontsize)
    ax4.set_title("去噪过程中新生成架构的唯一率变化", fontsize=title_fontsize)

    # 保存图片
    plt.savefig(f"{save_dir}/{dataset}_seed_{seed}.png", bbox_inches="tight")
    plt.close()
