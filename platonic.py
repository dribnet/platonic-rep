import os 
import torch

from datasets import load_dataset
from measure_alignment import compute_score, prepare_features


# NOTE: there are models we did not list, feel free to add more or your custom models
# here is the full list of precomputed features http://vision14.csail.mit.edu/prh/wit_1024/

SUPPORTED_DATASETS = {
    "wit_1024": {
        "dinov2_s": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_small_patch14_dinov2.lvd142m_pool-none.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_small_patch14_dinov2.lvd142m_pool-none.pt"
        },
        "dinov2_m": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_base_patch14_dinov2.lvd142m_pool-none.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_base_patch14_dinov2.lvd142m_pool-none.pt"
        },
        "dinov2_l": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_large_patch14_dinov2.lvd142m_pool-none.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_large_patch14_dinov2.lvd142m_pool-none.pt"
        },
        "dinov2_g": {
             "path": "./results/features/minhuh/prh/wit_1024/vit_giant_patch14_dinov2.lvd142m_pool-none.pt",
             "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_giant_patch14_dinov2.lvd142m_pool-none.pt"
        },
        "clip_b": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_base_patch16_clip_224.laion2b_pool-none.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_base_patch16_clip_224.laion2b_pool-none.pt"
        },
        "clip_l": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_large_patch14_clip_224.laion2b_pool-none.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_large_patch14_clip_224.laion2b_pool-none.pt"
        },
        "clip_h": {
            "path": "./results/features/minhuh/prh/wit_1024/vit_huge_patch14_clip_224.laion2b_pool-none.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/vit_huge_patch14_clip_224.laion2b_pool-none.pt"   
        },
        "llama-7b": {
            "path": "./results/features/minhuh/prh/wit_1024/huggyllama_llama-7b_pool-avg.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/huggyllama_llama-7b_pool-avg.pt"
        },
        "llama-13b": {
            "path": "./results/features/minhuh/prh/wit_1024/huggyllama_llama-13b_pool-avg.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/huggyllama_llama-13b_pool-avg.pt"
        },
        "llama-30b": { # NOTE this is 33B https://huggingface.co/huggyllama/llama-30b
            "path": "./results/features/minhuh/prh/wit_1024/huggyllama_llama-30b_pool-avg.pt",
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/huggyllama_llama-30b_pool-avg.pt"
        },
        "llama_65b": {
            "path": "./results/features/minhuh/prh/wit_1024/huggyllama_llama-65b_pool-avg.pt",   
            "url": "http://vision14.csail.mit.edu/prh/wit_1024/huggyllama_llama-65b_pool-avg.pt"
        }
    }
}


class Alignment():

    def __init__(self, dataset, subset, models=[], device="cuda", dtype=torch.bfloat16):
        
        if dataset != "minhuh/prh":
            # TODO: support external datasets in the future
            raise ValueError(f"dataset {dataset} not supported")
            
        if subset not in SUPPORTED_DATASETS:
            raise ValueError(f"subset {subset} not supported for dataset {dataset}")
        
        self.models = models
        self.device = device
        self.dtype = dtype

        # loads the features from path if it does not exist it will download
        self.features = {}
        for m in models:
            feat_path = SUPPORTED_DATASETS[subset][m]["path"]
            feat_url = SUPPORTED_DATASETS[subset][m]["url"]
            
            if not os.path.exists(feat_path):
                print(f"downloading features for {m} in {dataset}/{subset} from {feat_url}")
                
                # download and save the features in the feat_path
                os.makedirs(os.path.dirname(feat_path), exist_ok=True)
                os.system(f"wget {feat_url} -O {feat_path}")

                if not os.path.exists(feat_path):            
                    raise ValueError(f"feature path {feat_path} does not exist for {m} in {dataset}/{subset}")

            self.features[m] = self.load_features(feat_path)
            
        # download dataset from huggingface        
        self.dataset = load_dataset(dataset, revision=subset, split='train')
        return
    

    def load_features(self, feat_path):
        """ loads features for a model """
        return torch.load(feat_path, map_location=self.device)["feats"].to(dtype=self.dtype)

    
    def get_data(self, modality):
        """ load data 
        TODO: use multiprocessing to speed up loading
        """
        if modality == "text": # list of strings
            return [x['text'][0] for x in self.dataset]
        elif modality == "image": # list of PIL images
            return [x['image'] for x in self.dataset]
        else:
            raise ValueError(f"modality {modality} not supported")
    
    def score(self, features, metric, *args, **kwargs):
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
        scores = {}
        for m in self.models:
            scores[m] = compute_score(
                prepare_features(features.to(device=self.device, dtype=self.dtype)), 
                prepare_features(self.features[m].to(device=self.device, dtype=self.dtype)),
                metric, 
                *args, 
                **kwargs
            )
        return scores        
    
    