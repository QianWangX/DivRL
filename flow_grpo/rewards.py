from PIL import Image
import io
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew/500, meta

    return _fn

def aesthetic_score():
    from flow_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn

def clip_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def dino_score(device):
    from flow_grpo.dino_scorer import DinoScorer

    scorer = DinoScorer(device=device)

    def _fn(images, ref_images):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        if not isinstance(ref_images, torch.Tensor):
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8)/255.0
        scores = scorer(images, ref_images)
        return scores, {}

    return _fn


def image_similarity_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device).cuda()

    def _fn(images, ref_images):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        if not isinstance(ref_images, torch.Tensor):
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8)/255.0
        scores = scorer.image_similarity(images, ref_images)
        return scores, {}

    return _fn



def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def imagereward_score(device):
    from flow_grpo.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def qwenvl_score(device):
    from flow_grpo.qwenvl import QwenVLScorer

    scorer = QwenVLScorer(dtype=torch.bfloat16, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

    
def ocr_score(device):
    from flow_grpo.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def video_ocr_score(device):
    from flow_grpo.ocr import OcrScorer_video_or_image

    scorer = OcrScorer_video_or_image()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1) 
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def mtg_score_remote(device):
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import time
    import subprocess
    import pickle
    import base64
    import os
    import re
    
    # Force 'requests' to bypass proxies for local cluster addresses
    os.environ['no_proxy'] = 'localhost,127.0.0.1,' + os.getenv("REWARD_NODE_IP", "")
    os.environ['NO_PROXY'] = os.environ['no_proxy']
    
    def get_slurm_node_ip(node_name):
        if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", node_name):
            return node_name
        
        # If it's a name (like gpu201), try to resolve it
        try:
            cmd = f"scontrol show node {node_name} | grep NodeAddr"
            result = subprocess.check_output(cmd, shell=True).decode()
            return result.split('=')[1].strip().split()[0]
        except subprocess.CalledProcessError:
            # Fallback: if scontrol fails, just try using the input directly
            return node_name

    # 1. Configuration
    # REWARD_IP = os.environ.get("REWARD_NODE_IP", "localhost")  # SLURM node name where the MTG server is running
    # node_ip = get_slurm_node_ip(REWARD_IP)  # Replace with your SLURM node name if different
    import socket
    reward_ip = os.environ.get("REWARD_NODE_IP")
    if not reward_ip:
        # Fallback for local dev
        node_ip = "127.0.0.1"
    else:
        # Resolve hostname to IP safely without calling scontrol
        try:
            node_ip = socket.gethostbyname(reward_ip)
        except:
            node_ip = reward_ip
    REWARD_PORT = os.getenv("REWARD_PORT", 8098)
    url = f"http://{node_ip}:{REWARD_PORT}/score"
    print("MTG Reward Server URL:", url)
    
    
    batch_size = 32
    
    def _fn(images, prompts, metadata, ref_images, ref_image_masks=None, is_ssm=False,
            semantic_threshold=0.7):
        del prompts
        
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            if ref_images is not None:
                ref_images = [np.array(img) for img in ref_images]
                ref_images = np.array(ref_images)
            else:
                ref_images = images
                
            if ref_image_masks is not None:
                ref_image_masks = [np.array(mask) for mask in ref_image_masks]
                ref_image_masks = np.array(ref_image_masks)
            else:
                ref_image_masks = None
                
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        ref_images_batched = np.array_split(ref_images, np.ceil(len(ref_images) / batch_size))
        ref_image_masks_batched = None
        if ref_image_masks is not None:
            ref_image_masks_batched = np.array_split(ref_image_masks, np.ceil(len(ref_image_masks) / batch_size))
            
        all_scores = []
        for image_batch, ref_image_batch, ref_image_mask_batch in zip(images_batched, ref_images_batched, ref_image_masks_batched if ref_image_masks_batched is not None else [None]*len(images_batched)):
            png_images = []
            ref_png_images = []
            ref_png_image_masks = []

            # Compress the images using PNG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                png_images.append(img_str)
                
            for ref_image in ref_image_batch:
                ref_img = Image.fromarray(ref_image)
                ref_buffer = BytesIO()
                ref_img.save(ref_buffer, format="PNG")
                ref_img_str = base64.b64encode(ref_buffer.getvalue()).decode("utf-8")
                ref_png_images.append(ref_img_str)
                
            if ref_image_mask_batch is not None:
                for ref_image_mask in ref_image_mask_batch:
                    ref_mask_img = Image.fromarray(ref_image_mask)
                    ref_mask_buffer = BytesIO()
                    ref_mask_img.save(ref_mask_buffer, format="PNG")
                    ref_mask_img_str = base64.b64encode(ref_mask_buffer.getvalue()).decode("utf-8")
                    ref_png_image_masks.append(ref_mask_img_str)

            # format for mtg server
            if len(ref_png_image_masks) > 0:
                payload = {
                    "image_1": png_images,
                    "image_2": ref_png_images,
                    "image_2_object_mask": ref_png_image_masks,
                    "is_ssm": is_ssm,
                    "semantic_threshold": semantic_threshold,
                }
            else:
                payload = {
                    "image_1": png_images,
                    "image_2": ref_png_images,
                    "is_ssm": is_ssm,   
                    "semantic_threshold": semantic_threshold,
                }
            try:
                response = requests.post(url, json=payload)
                
                # Check status
                if response.status_code == 200:
                    result = response.json()
                    score = result["score"]
                    success_signal = [1.0] * len(image_batch)
                    
                else:
                    print("\n❌ ERROR")
                    print(response.text)
                    score = [0.0] * len(image_batch)
                    success_signal = [0.0] * len(image_batch)
                    
            except requests.exceptions.ConnectionError:
                print("\n❌ CONNECTION REFUSED")
                print("Is the server running? Check Terminal 1.")
                score = [0.0] * len(image_batch)
                
                success_signal = [0.0] * len(image_batch)

            all_scores += score

        return all_scores, {"success_signal": success_signal}
    
    return _fn
        
def geneval_score(device):
    """Submits images to GenEval and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn


def multi_score(device, score_dict, is_hinge=False, hinge_dict=None,
                semantic_threshold=0.7):
    score_functions = {
        "ocr": ocr_score,
        "video_ocr": video_ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "qwenvl": qwenvl_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "dinoscore": dino_score,
        "image_similarity": image_similarity_score,
        "mtgscore_vsm": mtg_score_remote,
        "mtgscore_nssm": mtg_score_remote,
    }
    score_fns={}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, ref_images=None, ref_image_masks=None, only_strict=True,
            is_hinge=is_hinge, hinge_dict=hinge_dict, semantic_threshold=semantic_threshold):
        total_scores = []
        score_details = {}
        
        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](images, prompts, metadata, only_strict)
                score_details['accuracy'] = rewards
                score_details['strict_accuracy'] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f'{key}_strict_accuracy'] = value
                for key, value in group_rewards.items():
                    score_details[f'{key}_accuracy'] = value
            elif score_name == "image_similarity":
                scores, rewards = score_fns[score_name](images, ref_images)
            elif score_name == "dinoscore":
                scores, rewards = score_fns[score_name](images, ref_images)
            elif score_name == "mtgscore_vsm":
                scores, rewards = score_fns[score_name](images, prompts, metadata, ref_images, ref_image_masks, semantic_threshold=semantic_threshold)
            elif score_name == "mtgscore_nssm":
                scores, rewards = score_fns[score_name](images, prompts, metadata, ref_images, ref_image_masks, is_ssm=True)
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            
            if not is_hinge:
                if isinstance(scores, list):
                    scores_tensor = torch.stack([torch.as_tensor(s) for s in scores]).to(device)
                    weighted_scores = weight * scores_tensor
                else:
                    weighted_scores = weight * scores.to(device)
            
            if not is_hinge:
                if len(total_scores) == 0:
                    total_scores = weighted_scores
                else:
                    total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        if not is_hinge:
            score_details['avg'] = total_scores
        if is_hinge:
            hinge_margin = hinge_dict.get("hinge_margin", 0.6)
            hinge_weight = hinge_dict.get("hinge_weight", 5.0)
            hinge_main = hinge_dict.get("hinge_main", "mtgscore_nssm")
            hinge_gate = hinge_dict.get("hinge_gate", "mtgscore_vsm")
            assert hinge_main in score_details, f"Hinge main score '{hinge_main}' not found in score details."
            assert hinge_gate in score_details, f"Hinge gate score '{hinge_gate}' not found in score details."
            
            # Convert to Python lists before iterating so CUDA tensors can be freed
            if isinstance(score_details[hinge_main], torch.Tensor):
                main_scores = score_details[hinge_main].detach().cpu().tolist()
                score_details[hinge_main] = main_scores   # overwrite in-place with Python list
            else:
                main_scores = score_details[hinge_main]
            
            gate_scores = score_details[hinge_gate]
            if isinstance(gate_scores, torch.Tensor):
                gate_scores = gate_scores.detach().cpu().tolist()
            
            hinge_scores = []
            for main_score, gate_score in zip(main_scores, gate_scores):
                if gate_score < hinge_margin:
                    hinge_score = main_score - hinge_weight * (hinge_margin - gate_score) ** 2
                else:
                    hinge_score = main_score
                hinge_scores.append(torch.as_tensor(hinge_score, device=device))
            score_details['avg'] = hinge_scores

        return score_details, {}

    return _fn

def main():
    import torchvision.transforms as transforms

    image_paths = [
        "nasa.jpg",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {
        "unifiedreward": 1.0
    }
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    scores, _ = scoring_fn(images, prompts, metadata)


if __name__ == "__main__":
    main()
