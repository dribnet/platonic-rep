import platonic
from models import load_llm, load_tokenizer
from tqdm.auto import trange
import torch 
from pprint import pprint
from measure_alignment import compute_score, prepare_features
from metrics import compute_nearest_neighbors
import torch.nn.functional as F

import csv
import sys

texts = []
with open("5000-more-common.txt", "r", newline="") as file:
    # skip header
    next(file)
    texts = [line.rstrip() for line in file]

# print(f"there are {len(lines)} lines, here are the first 5")
# print(lines[:5])
# sys.exit(0)

device = "cuda"
dtype=torch.float

# your model (e.g. we will use open_llama_7b as an example)
# model_name = "openlm-research/open_llama_7b"
# model_name1 = "bert-base-cased"
model_name1 = "mixedbread-ai/mxbai-embed-large-v1"
# model_name2 = "bert-base-uncased"
model_name2 = "mixedbread-ai/mxbai-embed-2d-large-v1"
model_names = [model_name1, model_name2]

# https://stackoverflow.com/a/68525048

model_results = []
def diff_all_rows(x): 
   y = x[None] - x[:, None] 
   ind = torch.tril_indices(x.shape[0], x.shape[0], offset=-1)
   return y[ind[0], ind[1]]

for model_name in model_names:
    language_model = load_llm(model_name, qlora=False)
    tokenizer = load_tokenizer(model_name)

    # extract features
    tokens = tokenizer(texts, padding="longest", return_tensors="pt")        
    print("some tokens", tokens[0])

    batch_size = 16
    feature_pairs = []

    llm_feats = []
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
    model_results.append(llm_feats)

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

print(f"LLM {model_names[0]} FEATS SIZE IS: ", model_results[0].shape)
print(f"LLM {model_names[1]} FEATS SIZE IS: ", model_results[1].shape)

build_standard_score(model_results[0], model_results[1], texts)
# build_standard_score(model_results[0], model_results[1], texts)

# compute score
# score = platonic_metric.score(llm_feats, metric="mutual_knn", topk=10, normalize=True)


# score2 = score_pairs(feature_pairs[1], feature_pairs[0])

# print(score1, score2) # it will print the score and the index of the layer the maximal alignment happened
