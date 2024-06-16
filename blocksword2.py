import platonic
from models import load_llm, load_tokenizer
from tqdm.auto import trange
import torch 
from pprint import pprint
from measure_alignment import compute_score, prepare_features

import csv
import sys

word_pairs = []
with open("wordpairs1.csv", "r", newline="") as file:
    reader = csv.reader(file, delimiter=",")
    # skip header
    next(reader)
    for row in reader:
        if(len(word_pairs) < 100):
            word_pairs.append([ row[0], row[1] ])

texts1 = [w[0] for w in word_pairs]
texts2 = [w[1] for w in word_pairs]
print("some words", texts2[:5])
# sys.exit(0)

device = "cuda"
dtype=torch.float

# your model (e.g. we will use open_llama_7b as an example)
# model_name = "openlm-research/open_llama_7b"
# model_name1 = "bert-base-cased"
model_name1 = "mixedbread-ai/mxbai-embed-large-v1"
# model_name1 = "bert-base-uncased"
model_name2 = "mixedbread-ai/mxbai-embed-2d-large-v1"
model_names = [model_name1, model_name2]

# model_name = "mixedbread-ai/mxbai-embed-2d-large-v1"

# https://stackoverflow.com/a/68525048

model_result_pairs = []
def diff_all_rows(x): 
   y = x[None] - x[:, None] 
   ind = torch.tril_indices(x.shape[0], x.shape[0], offset=-1)
   return y[ind[0], ind[1]]

for model_name in model_names:
    language_model = load_llm(model_name, qlora=False)
    tokenizer = load_tokenizer(model_name)

    # extract features
    tokens1 = tokenizer(texts1, padding="longest", return_tensors="pt")        
    tokens2 = tokenizer(texts2, padding="longest", return_tensors="pt")        
    print("some tokens", tokens1[0], tokens2[0])
    token_pairs = [tokens1, tokens2]

    batch_size = 16
    feature_pairs = []

    for p in range(2):
        tokens = token_pairs[p]
        llm_feats = []
        for i in trange(0, len(word_pairs), batch_size):
            token_inputs = {k: v[i:i+batch_size].to(device).long() for (k, v) in tokens.items()}
            with torch.no_grad():
                llm_output = language_model(
                    input_ids=token_inputs["input_ids"],
                    attention_mask=token_inputs["attention_mask"],
                )
            feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3).cpu()
            mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1).cpu()
            feats = (feats * mask).sum(2) / mask.sum(2)
            llm_feats.append(feats)
            # import ipdb; ipdb.set_trace()
        llm_feats = torch.cat(llm_feats)
        # all_feats = llm_feats
        llm_feats2 = diff_all_rows(llm_feats)
        all_feats = torch.cat([llm_feats,llm_feats2[:5000]])
        print("all_feats shape ", all_feats.shape)
        # llm_feats2=llm_feats2[torch.randperm(llm_feats2.size()[0])][:500]
        # llm_feats3 = diff_all_rows(llm_feats2)
        # print("feats2 -> 3 shape ", llm_feats2.shape, llm_feats3.shape)
        # llm_feats3=llm_feats3[torch.randperm(llm_feats3.size()[0])][:500]
        # print("feats3 final ", llm_feats3.shape)
        feature_pairs.append(all_feats)
    model_result_pairs.append(feature_pairs)

def score_pairs(features1, features2):
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
        prepare_features(features1, exact=True).to(dtype=dtype), 
        prepare_features(features2, exact=True).to(dtype=dtype),
        "mutual_knn", topk=4, normalize=True, only_layer=-1)
    return result

for m_i in range(len(model_names)):
    model_name_main = model_names[m_i]
    feature_pairs_main = model_result_pairs[m_i]
    print(f"LLM {model_name_main} FEATS SIZE IS: ", feature_pairs_main[0].shape, feature_pairs_main[1].shape)

    score1 = score_pairs(feature_pairs_main[0], feature_pairs_main[1])
    print(f"score {model_name_main} 1->2 {score1}")

    for m_j in range(m_i+1, len(model_names)):
        model_name_other = model_names[m_j]
        feature_pairs_other = model_result_pairs[m_j]
        print(f"LLM {model_name_other} FEATS SIZE IS: ", feature_pairs_other[0].shape, feature_pairs_other[1].shape)

        score2 = score_pairs(feature_pairs_main[0], feature_pairs_other[0])
        print(f"score {model_name_main}->{model_name_other} 1->1 {score2}")

        score2 = score_pairs(feature_pairs_main[1], feature_pairs_other[1])
        print(f"score {model_name_main}->{model_name_other} 2->2 {score2}")

        score2 = score_pairs(feature_pairs_main[0], feature_pairs_other[1])
        print(f"score {model_name_main}->{model_name_other} 1->2 {score2}")

        score2 = score_pairs(feature_pairs_main[1], feature_pairs_other[0])
        print(f"score {model_name_main}->{model_name_other} 2->1 {score2}")

# compute score
# score = platonic_metric.score(llm_feats, metric="mutual_knn", topk=10, normalize=True)


# score2 = score_pairs(feature_pairs[1], feature_pairs[0])

# print(score1, score2) # it will print the score and the index of the layer the maximal alignment happened
