import torch
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from configs.config import load_dataset_generation_config

from safetensors.torch import load_file
from torchvision import transforms
import torch

import mtg.models as mtg_models
from mtg.utils.aux_models_manager import ModelManager

from datasets import load_dataset

import os
import sys
sys.path.append('..')
from pathlib import Path
import numpy as np
from cal_ssm import compute_masked_nssm

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

class MTGScorer(torch.nn.Module):
    def __init__(self, device="cuda", seed=0):
        super().__init__()
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        mtg_ckpt_pth = Path(hf_hub_download(repo_id="abdo-eldesokey/mind-the-glitch", filename="mtg_weights.safetensors"))
        mtg_config = Path(hf_hub_download(repo_id="abdo-eldesokey/mind-the-glitch", filename="experiment_cfg.yaml"))
        training_cfg = OmegaConf.load(mtg_config)
        dataset_cfg = load_dataset_generation_config("automated")

        # Define constants from config
        DTYPE = getattr(torch, training_cfg.dtype, "float32")
        MODEL_NAME = training_cfg.model.name
        
        self.device = device

        # Load Auxiliary models
        models_manager = ModelManager(device, dataset_cfg ,load_cleandift=True, load_groundingsam=False)
        cleandift_model = models_manager.cleandift_model

        self.transform = transforms.Compose([
                                        transforms.Resize(dataset_cfg.img_size), 
                                        transforms.ToTensor()])    

        # Load MTG Model
        self.model = getattr(mtg_models, MODEL_NAME)(cleandift_model, training_cfg, device, DTYPE, seed=seed)
        print(f"Loading checkpoint from {mtg_ckpt_pth}")
        if mtg_ckpt_pth.suffix == ".safetensors":
            self.model.load_state_dict(load_file(mtg_ckpt_pth))
        else:
            self.model.load_state_dict(torch.load(mtg_ckpt_pth, map_location=device))
        self.model.to(device)
        self.model.eval()

        
    @torch.no_grad()
    def __call__(self, img1, img2, img1_obj_mask=None, img2_obj_mask=None, is_ssm=False, semantic_threshold=0.7):
        if is_ssm:
            model_out_dict = self.model.inference(
                                                img1, img2, 
                                                img1_obj_mask_p=None, 
                                                img2_obj_mask_p=img2_obj_mask, 
                                                img1_part_mask_p=None, 
                                                img2_part_mask_p=None, 
                                                transform=self.transform, 
                                                return_correspondences=True,
                                                semantic_threshold=semantic_threshold,)
            
            gen_feat, ref_feat = model_out_dict["img1_feats_v"], model_out_dict["img2_feats_v"]
            gen_feat = torch.from_numpy(gen_feat).to(self.device)
            ref_feat = torch.from_numpy(ref_feat).to(self.device)           
            reward = compute_masked_nssm(gen_feat, ref_feat, img2_obj_mask)

        else:
            model_out_dict = self.model.inference(
                                                img1, img2, 
                                                img1_obj_mask_p=img1_obj_mask, 
                                                img2_obj_mask_p=img2_obj_mask, 
                                                img1_part_mask_p=None, 
                                                img2_part_mask_p=None, 
                                                transform=self.transform, 
                                                return_correspondences=False,
                                                semantic_threshold=semantic_threshold,)
            metrics = model_out_dict["metrics"]
            reward = metrics["vsm_0.7"]

        return reward
