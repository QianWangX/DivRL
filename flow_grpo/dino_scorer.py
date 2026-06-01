import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image

class DinoScorer(nn.Module):
    def __init__(self, model_type="dinov2_vits14", device="cuda"):
        super().__init__()
        self.device = device
        self.model = torch.hub.load('facebookresearch/dinov2', model_type).to(device)
        self.model.eval()
        
        # 1. Preprocessing for PIL images
        self.pil_transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(), # Converts PIL 0-255 to Tensor 0.0-1.0
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 2. Preprocessing for Tensors
        # Note: Omit ToTensor(). antialias=True is highly recommended when resizing tensors.
        self.tensor_transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def _preprocess(self, images):
        """Converts PIL images or Tensors into a single preprocessed batched tensor."""
        model_dtype = next(self.model.parameters()).dtype
        
        # A. Handle a single Tensor [B, C, H, W] or unbatched [C, H, W]
        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = images.unsqueeze(0) # Add batch dimension if missing
            
            # If the tensor is uint8 (0-255), safely convert to float (0.0-1.0)
            if images.dtype == torch.uint8:
                images = images.to(dtype=model_dtype) / 255.0
            else:
                # Force cast to match model weights (catches float64, float16, etc.)
                images = images.to(dtype=model_dtype)
                
            images = images.to(self.device)
            return self.tensor_transform(images)

        # B. Handle a Python list of items
        elif isinstance(images, list):
            if len(images) == 0:
                raise ValueError("Empty image list provided.")

            # List of PIL Images
            if isinstance(images[0], Image.Image):
                tensors = [self.pil_transform(img.convert("RGB")) for img in images]
                return torch.stack(tensors).to(self.device)
            
            # List of PyTorch Tensors
            elif isinstance(images[0], torch.Tensor):
                processed_tensors = []
                for t in images:
                    if t.dtype == torch.uint8:
                        t = t.float() / 255.0
                    t = t.to(self.device)
                    processed_tensors.append(self.tensor_transform(t))
                return torch.stack(processed_tensors)
            
            else:
                raise TypeError("List elements must be PIL Images or PyTorch Tensors.")
        else:
            raise TypeError("Input must be a list of images or a batched PyTorch Tensor.")

    @torch.no_grad()
    def get_embeddings(self, images):
        """Returns normalized embeddings for images (PIL list, Tensor list, or batched Tensor)."""
        pixel_values = self._preprocess(images)
        embeddings = self.model(pixel_values)
        # L2 Normalization makes dot product equivalent to Cosine Similarity
        return F.normalize(embeddings, p=2, dim=-1)

    @torch.no_grad()
    def __call__(self, input_a, input_b):
        """
        Calculates cosine similarity between two inputs.
        Inputs can be lists of PIL images, lists of Tensors, or batched Tensors.
        """
        # Dynamically check length based on whether input is a list or a batched tensor
        len_a = len(input_a) if isinstance(input_a, list) else input_a.shape[0]
        len_b = len(input_b) if isinstance(input_b, list) else input_b.shape[0]
        
        if len_a != len_b:
            raise ValueError(f"Inputs must have the same batch size. Got {len_a} and {len_b}.")

        embeds1 = self.get_embeddings(input_a)
        embeds2 = self.get_embeddings(input_b)
        
        # Calculate row-wise dot product (Cosine Similarity)
        similarity = (embeds1 * embeds2).sum(dim=-1)
        return similarity

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = DinoScorer(device=device)

    # --- Test 1: PIL Images ---
    # imgs_a = [Image.open("image1.jpg")] ...
    
    # --- Test 2: Tensors ---
    # Create dummy tensors representing batched RGB images [B, C, H, W]
    tensor_batch_a = torch.rand(2, 3, 512, 512) 
    tensor_batch_b = torch.rand(2, 3, 300, 400) 

    scores = scorer(tensor_batch_a, tensor_batch_b)
    
    for i, s in enumerate(scores):
        print(f"Pair {i} Similarity: {s.item():.4f}")

if __name__ == "__main__":
    main()