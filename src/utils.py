import matplotlib.pyplot as plt
import random
import numpy as np
import os
import torch
import cv2

# fix seeds
def seed_everything_custom(seed: int = 91):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # gpu vars
    torch.backends.cudnn.deterministic = True  #needed
    torch.backends.cudnn.benchmark = False    
    # seed_everything(seed)
    print(f"seed{seed}, applied")

def show_result_image(result, num=None):
    norm_info = [result["pred_masks"].max(), result["pred_masks"].min()]
    fig_nums = result["pred_masks"].shape[0]
    if num is not None:
        fig_nums = num
    fig = plt.figure(figsize=(18, fig_nums*6))
    axs = fig.subplots(fig_nums, 3)
    for index, (path, anomaly_score, anomaly_label, pred_mask, gt) in enumerate(zip(
        result["image_paths"],
        result["anomaly_scores"],
        result["anomaly_labels"],
        result["pred_masks"],
        result["annotations"],
    )):
        axs[index][0].imshow(cv2.imread(path))
        axs[index][0].set_title(os.path.basename(path))
        mask = (pred_mask - norm_info[1])*255 / (norm_info[0] - norm_info[1])
        mask = np.stack([mask]*3, -1).astype(np.uint8)
        axs[index][1].imshow(gt)
        axs[index][1].set_title("ground truth")
        axs[index][2].imshow(mask)
        axs[index][2].set_title(f"anomaly_score:{anomaly_score:.2f}")
        if index == fig_nums-1:
            break

def show_accuracies_auroc(results):
    keys = []
    custom_accuracies = []
    custom_aurocs = []
    for key, r in results.items():
        keys.append(key)
        custom_accuracies.append(r["accuracy"])
        custom_aurocs.append(r["auroc"])
    fig = plt.figure(figsize=(18, 8))
    ax = fig.subplots(1,2)
    ax[0].bar(keys, custom_aurocs, alpha=0.8, color='red')
    ax[0].set_title("AUROC")
    
    ax[1].bar(keys, custom_accuracies, alpha=0.8, color='green')
    ax[1].set_title("ACCURACY")

def show_score_distribution(result):
    fig = plt.figure(figsize=(18, 8))
    plt.hist(result["anomaly_scores"][result["anomaly_labels"] == True], color="red", bins=50)
    plt.hist(result["anomaly_scores"][result["anomaly_labels"] == False], color="green", bins=50)
    plt.legend(["anomaly", "normal"])
    plt.xlabel("anomaly score")
    plt.ylabel("sample num")
    th = result["th"]
    plt.title(f"Threshold value:{th:.2f} anomaly score distribution")
    plt.show()