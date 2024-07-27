import platonic
import metrics
import argparse 
from models import load_llm, load_tokenizer
from tqdm.auto import trange
import torch 
from pprint import pprint
from measure_alignment import compute_score, prepare_features
from metrics import compute_nearest_neighbors
import torch.nn.functional as F
import numpy as np
from numpy import genfromtxt

import csv
import sys
import matplotlib
from matplotlib import pyplot as plt

device = "cpu"
dtype=torch.float


ONLY_LAYER = 0

def score_pairs(features1, features2, topk):
    """ 
    Args:
        features (torch.Tensor): features to compare
        metric (str): metric to use
        *args: additional arguments for compute_score / metrics.AlignmentMetrics
        **kwargs: additional keyword arguments for compute_score / metrics.AlignmentMetrics
    Returns:
        dict: scores for each model organized as 
            {model_name: (score, layer_indices)} 
            layer_indices are the index of the layer with maximal alignment
    """
    result = compute_score(
        # prepare_features(features1, exact=True).to(device=device, dtype=dtype), 
        # prepare_features(features2, exact=True).to(device=device, dtype=dtype),
        # prepare_features(features1, exact=True).to(dtype=dtype), 
        # prepare_features(features2, exact=True).to(dtype=dtype),
        features1.to(dtype=dtype), 
        features2.to(dtype=dtype),
        "mutual_knn", topk=topk, normalize=True, only_layer=ONLY_LAYER)
    return result

def gimme_example(x, y, texts, topk):
    knn_A = compute_nearest_neighbors(x, topk)
    knn_B = compute_nearest_neighbors(y, topk)
    # print(f"knn results {knn_A.shape}, {knn_B.shape}")
    # n = knn_A.shape[0]
    # topk = knn_A.shape[1]
    s = f"A) {texts[0]} -> ("
    for n in knn_A[0]:
        print(f"fetchin {n}")
        s = s + f"{texts[n]},"
    s = s + ")"
    print(s)
    s = f"B) {texts[0]} -> ("
    for n in knn_B[0]:
        print(f"fetchin {n}")
        s = s + f"{texts[n]},"
    s = s + ")"
    print(s)
    # print(f"with an input of {texts[0]} the indexA is maybe {knn_A[0]}")
    # print(f"with an input of {texts[0]} the indexB is maybe {knn_B[0]}")

def select_random_subset(features1, features2, texts, bs):
    num_features = features1.shape[0]
    index = torch.randperm(num_features)[:bs]
    print("random subset starts with ", index[:5])
    sub_features1 = features1[index]
    sub_features2 = features2[index]
    # flat list version
    sub_texts = [texts[i] for i in index]
    # sub_texts = texts[index]
    return [sub_features1, sub_features2, sub_texts]

def select_random_subset_d1(features1, features2, texts, bs):
    num_features = features1.shape[0]
    indexA = torch.randperm(num_features)[:bs]
    indexB = torch.randperm(num_features)[:bs]
    print("random subset starts with ", indexA[:5], indexB[:5])
    sub_features1 = features1[indexA] - features1[indexB]
    sub_features2 = features2[indexA] - features2[indexB]
    # flat list version
    sub_texts = [f"{texts[indexA[i]]}:{texts[indexB[i]]}" for i in range(bs)]
    # sub_texts = texts[index]
    return [sub_features1, sub_features2, sub_texts]

def build_standard_score(features1, features2, texts):
    # for bs in [4000, 2000, 1000, 500, 200]:
    # for bs in [4000, 1000, 200]:
    for bs in [500, 5000]:
        [f1, f2, t] = select_random_subset(features1, features2, texts, bs)
        # for i in [1, 2, 5, 10, 20]:
        # for i in [1, 2, 5, 10, 20, 50]:
        for i in [1, 5]:
            if i < 10:
                gimme_example(f1, f2, t, i)
            s = score_pairs(f1, f2, i)
            print(f"score s={bs}, k={i}: {s}")

# print(f"LLM {model_names[0]} FEATS SIZE IS: ", model_results[0].shape)
# print(f"LLM {model_names[1]} FEATS SIZE IS: ", model_results[1].shape)

# build_standard_score(model_results[0], model_results[1], texts)

# https://github.com/Cysu/open-reid/commit/61f9c4a4da95d0afc3634180eee3b65e38c54a14
def validate_cov_matrix(M):
    M = (M + M.T) * 0.5
    k = 0
    I = np.eye(M.shape[0])
    while True:
        try:
            _ = np.linalg.cholesky(M)
            break
        except np.linalg.LinAlgError:
            # Find the nearest positive definite matrix for M. Modified from
            # http://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
            # Might take several minutes
            k += 1
            w, v = np.linalg.eig(M)
            min_eig = v.min()
            M += (-min_eig * k * k + np.spacing(min_eig)) * I
    return M

# https://gist.github.com/mdouze/6cc12fa967e5d9911580ef633e559476
def mahalnobis_to_L2(x_torch):
    return x_torch

    # print(f"input is {x_torch.shape}")
    x = x_torch.cpu().detach().numpy()
    x = np.swapaxes(x,0,1)
    # compute and visualize the covariance matrix
    xc = x - x.mean(0)
    cov = np.dot(xc.T, xc) / xc.shape[0]
    # print(f"cov is {cov}")
    plt.imsave("before.png", cov)

    # map the vectors back to a space where they follow a unit Gaussian
    cov = validate_cov_matrix(cov)
    L = np.linalg.cholesky(cov)
    mahalanobis_transform = np.linalg.inv(L)
    y = np.dot(x, mahalanobis_transform.T)

    # covariance should be diagonal in that space...
    yc = y - y.mean(0)
    ycov = np.dot(yc.T, yc) / yc.shape[0]
    plt.imsave("after.png", ycov)

    y = np.swapaxes(y,0,1)

    return(torch.tensor(y))

def nobis_score(features1, features2, topk):
    # metric = "mutual_knn"
    x = mahalnobis_to_L2(features1)
    y = mahalnobis_to_L2(features2)
    # print(f"AFTER_SHAPES: {x.shape}, {y.shape}, {topk}")
    cur_s = metrics.AlignmentMetrics.mutual_knn(x, y, topk=topk)
    # cur_s = metrics.AlignmentMetrics.measure("mutual_knn", x, y, topk=topk)
    return cur_s

if __name__ == "__main__":
    """
    """
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--labels",     type=str, default="inputs/pipe_all_train_filtered.txt")
    # parser.add_argument("--v1",         type=str, default="inputs/lpipe_train1_2_train_points_label0.csv")
    # parser.add_argument("--v1",         type=str, default="inputs/vectors_pipe_train_glm3.tsv")
    # parser.add_argument("--v2",         type=str, default="inputs/vectors_pipe_train_llama3.tsv")
    parser.add_argument("--v1",         type=str, default="inputs/vectors_pipe_train.tsv")
    parser.add_argument("--v2",         type=str, default="inputs/vectors_pipe_train_sae_l11.tsv")

    # parser.add_argument("--labels",     type=str, default="inputs/pipe_all_test_filtered.txt")
    # parser.add_argument("--v1",         type=str, default="inputs/lpipe_train1_2_test1_points_label0.csv")
    # parser.add_argument("--v2",         type=str, default="inputs/vectors_pipe_test.tsv")
    # # # parser.add_argument("--v1",         type=str, default="inputs/vectors_pipe_test_angle.tsv")

    # let's test bang 5k train/test map and then maybe try a repeat with mutual knn metric
    # parser.add_argument("--labels",     type=str, default="inputs/bang_all_train_filtered.txt")
    # parser.add_argument("--v1",         type=str, default="inputs/bang_train1_0_train1_points_label0.csv")
    # parser.add_argument("--v2",         type=str, default="inputs/vectors_bang_train.tsv")

    # parser.add_argument("--v1",         type=str, default="inputs/bang_train1_0_test1_points_label0.csv")

    args = parser.parse_args()
    

    print(f"reading labels from {args.labels}")
    with open(args.labels, "r", newline="") as file:
        # skip header
        texts = [line.rstrip() for line in file]

    print(f"reading vectors from {args.v1}")
    delim = '\t'
    if args.v1.endswith("csv"):
        delim = ','
    v1 = genfromtxt(args.v1, delimiter=delim, max_rows=None)
    print(f"OLD SHAPE {v1.shape}")
    lay = 11
    v1 = v1[:,(768*(lay)):(768*(lay+1))]
    print(f"NEW SHAPE {v1.shape}")
    x = torch.from_numpy(v1).to(dtype).to(device)
    if args.v1.endswith("csv"):
        x1, y1 = x.cpu().numpy().T
        x = F.normalize(x, p=2, dim=-1)
        x2, y2 = x.cpu().numpy().T
        plt.scatter(x1,y1)
        plt.savefig("vec1.png")
        plt.scatter(x2,y2)
        plt.savefig("vec2.png")

    print(f"reading vectors from {args.v2}")
    delim = '\t'
    if args.v2.endswith("csv"):
        delim = ','
    v2 = genfromtxt(args.v2, delimiter=delim, max_rows=None)
    y = torch.from_numpy(v2).to(dtype).to(device)
    # y = F.normalize(y, p=2, dim=-1)
    
    print(f"Shapes: {x.shape} {y.shape}")

    metric = "mutual_knn"

    # for bs in [500, 5000]:
    full_bs = len(texts)
    for bs in [2000, full_bs]:
    # for bs in [500, 1000, 2000, full_bs]:
    # for bs in [full_bs]:
        [f1, f2, t] = select_random_subset(x, y, texts, bs)
        # print(f"SUBSHAPES: {f1.shape}, {f2.shape}, {len(t)}")
        # for i in [1, 5, 20, 50]:
        # for i in [1, 5, 20, 50, 100, 500, 1000]:
        for p in [0.5, 1, 2, 5, 10, 20, 50]:
            i = int(bs * p / 100.0)
            # if i < 5:
            #     gimme_example(f1, f2, t, i)
            s = nobis_score(f1, f2, i)
            boost = s - (p / 100.0)
            # s = metrics.AlignmentMetrics.measure(metric, f1, f2, topk=i)
            print(f"score s={bs}, p={p}% k={i}: {s:4.2f}, boost={boost:4.2f}")

