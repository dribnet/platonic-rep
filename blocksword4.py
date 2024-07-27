# the purpose of this one is to test norm v no-norm and 
# part vs whole

import platonic
import metrics
from models import load_llm, load_tokenizer
from tqdm.auto import trange
import torch 
from pprint import pprint
from measure_alignment import compute_score, prepare_features
from metrics import compute_nearest_neighbors
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot

import csv
import sys
import os

texts = []
with open("5000-more-common.txt", "r", newline="") as file:
    # skip header
    next(file)
    texts = [line.rstrip() for line in file]

# print(f"there are {len(lines)} lines, here are the first 5")
# print(lines[:5])
# sys.exit(0)

device = "cpu"
dtype=torch.float

# your model (e.g. we will use open_llama_7b as an example)
# model_name = "openlm-research/open_llama_7b"
model_name1 = "bert-base-uncased"
# model_name1 = "mixedbread-ai/mxbai-embed-large-v1"
model_name2 = "bert-base-uncased"
# model_name2 = "bert-base-uncased"
# model_name2 = "mixedbread-ai/mxbai-embed-2d-large-v1"
model_names = [model_name1, model_name2]

# https://stackoverflow.com/a/68525048

model_results = []

for model_name in model_names:
    llm_feats = []
    outfile = f"outputs/vectors_{model_name}.npy"
    if os.path.exists(outfile):
        feats1 = np.load(outfile)
        llm_feats = torch.tensor(feats1)
    else:
        language_model = load_llm(model_name, qlora=False)
        tokenizer = load_tokenizer(model_name)

        # extract features
        tokens = tokenizer(texts, padding="longest", return_tensors="pt")        
        print("some tokens", tokens[0])

        batch_size = 16
        feature_pairs = []

        for i in trange(0, len(texts), batch_size):
            token_inputs = {k: v[i:i+batch_size].to(device).long() for (k, v) in tokens.items()}
            with torch.no_grad():
                llm_output = language_model(
                    input_ids=token_inputs["input_ids"],
                    attention_mask=token_inputs["attention_mask"],
                    output_hidden_states=True
                )
            feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3).cpu()
            mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1).cpu()
            feats = (feats * mask).sum(2) / mask.sum(2)
            llm_feats.append(feats)
            # import ipdb; ipdb.set_trace()
        llm_feats = torch.cat(llm_feats)
        feats1 = llm_feats.cpu().detach().numpy()
        np.save(outfile, feats1)

    model_results.append(llm_feats)

# https://stackoverflow.com/a/59187836
def to_positive_definitive(M):
    M = np.matrix(M)
    M = (M + M.T) * 0.5
    k = 1
    I = np.eye(M.shape[0])
    w, v = np.linalg.eig(M)
    min_eig = v.min()
    M += (-min_eig * k * k + np.spacing(min_eig)) * I
    return M

def validate_positive_definitive(M):   
    try:
        np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        print(f"matrix patch {M.shape}")
        M = to_positive_definitive(M)
    #Print the eigenvalues of the Matrix
    # print(np.linalg.eigvalsh(p))
    return M
# M = validate_positive_definitive(M)
# print(M)

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
    pyplot.imsave("before.png", cov)

    # map the vectors back to a space where they follow a unit Gaussian
    cov = validate_cov_matrix(cov)
    L = np.linalg.cholesky(cov)
    mahalanobis_transform = np.linalg.inv(L)
    y = np.dot(x, mahalanobis_transform.T)

    # covariance should be diagonal in that space...
    yc = y - y.mean(0)
    ycov = np.dot(yc.T, yc) / yc.shape[0]
    pyplot.imsave("after.png", ycov)

    y = np.swapaxes(y,0,1)

    return(torch.tensor(y))

def manual_score(features1, features2, topk, layer1, layer2, norm1, norm2):
    metric = "mutual_knn"
    if layer1 == -1:
        x = features1.flatten(1, 2).to(dtype=dtype)
    else:
        x = features1[:, layer1, :].to(dtype=dtype)
    x = mahalnobis_to_L2(x)
    if norm1:
        x = F.normalize(x, p=2, dim=-1)
    if layer2 == -1:
        y = features2.flatten(1, 2).to(dtype=dtype)
    else:
        y = features2[:, layer2, :].to(dtype=dtype)
    y = mahalnobis_to_L2(y)
    if norm2:
        y = F.normalize(y, p=2, dim=-1)
    # cur_s = metrics.AlignmentMetrics.measure(metric, x, y, topk=topk)
    cur_s = metrics.AlignmentMetrics.mutual_knn(x, y, topk=topk)
    return cur_s

def gimme_example(features1, features2, texts, topk):
    x = features1[:, ONLY_LAYER, :]
    x = F.normalize(x, p=2, dim=-1)
    y = features2[:, ONLY_LAYER, :]
    y = F.normalize(y, p=2, dim=-1)
    knn_A = compute_nearest_neighbors(x, topk)
    knn_B = compute_nearest_neighbors(y, topk)
    # print(f"knn results {knn_A.shape}, {knn_B.shape}")
    # n = knn_A.shape[0]
    # topk = knn_A.shape[1]
    s = f"A) {texts[0]} -> ("
    for n in knn_A[0]:
        s = s + f"{texts[n]},"
    s = s + ")"
    print(s)
    s = f"B) {texts[0]} -> ("
    for n in knn_B[0]:
        s = s + f"{texts[n]},"
    s = s + ")"
    print(s)
    # print(f"with an input of {texts[0]} the indexA is maybe {knn_A[0]}")
    # print(f"with an input of {texts[0]} the indexB is maybe {knn_B[0]}")

def select_random_subset(features1, features2, texts, bs):
    num_features = features1.shape[0]
    index = torch.randperm(num_features)[:bs]
    # print("random subset starts with ", index[:5])
    sub_features1 = features1[index]
    sub_features2 = features2[index]
    # flat list version
    sub_texts = [texts[i] for i in index]
    # sub_texts = texts[index]
    return [sub_features1, sub_features2, sub_texts]

def build_standard_score(features1, features2, texts):
    # features1 = torch.randn_like(features1)
    # features2 = torch.randn_like(features2)
    for bs in [1000, 5000]:
        [f1, f2, t] = select_random_subset(features1, features2, texts, bs)
        # for i in [1, 5, 20, 50]:
        for p in [0.25, 1, 5, 10, 20, 50]:
            i = int(bs * p / 100.0)
            s2 = manual_score(f1, f2, i, 1, 2, False, False)
            boost = s2 - (p / 100.0)
            # print(f"manual score s={bs}, k={i}: {s2}")
            print(f"score s={bs}, p={p}% k={i}: {s2:4.2f}, boost={boost:4.2f}")

print(f"LLM {model_names[0]} FEATS SIZE IS: ", model_results[0].shape)
print(f"LLM {model_names[1]} FEATS SIZE IS: ", model_results[1].shape)

build_standard_score(model_results[0], model_results[1], texts)
# build_standard_score(model_results[0], model_results[1], texts)

# compute score
# score = platonic_metric.score(llm_feats, metric="mutual_knn", topk=10, normalize=True)


# score2 = score_pairs(feature_pairs[1], feature_pairs[0])

# print(score1, score2) # it will print the score and the index of the layer the maximal alignment happened
