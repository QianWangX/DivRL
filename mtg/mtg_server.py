import torch
from fastapi import FastAPI, Request
import uvicorn
from PIL import Image
import io
import base64
import os
import numpy as np
import argparse

from mtg_score import MTGScorer

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

mtgscorer = MTGScorer(device=device)

@app.post("/score")
async def evaluate(data: dict):
    
    def to_pil_list(input_data, mode="RGB"):
        if input_data is None:
            return None
        
        # Ensure we are working with a list
        items = input_data if isinstance(input_data, list) else [input_data]
        
        pil_images = []
        for item in items:
            img_bytes = base64.b64decode(item)
            img = Image.open(io.BytesIO(img_bytes)).convert(mode)
            pil_images.append(img)
            
            # --- 2. Convert to Numpy for Value Validation ---
            # This is the most reliable way to check for NaNs or Infinity
            img_array = np.array(img)

            # Check for NaNs (Not a Number)
            # Note: Only relevant if the image is in a float mode (like 'F')
            if np.issubdtype(img_array.dtype, np.floating):
                if np.isnan(img_array).any():
                    print("❌ Validation Error: Image contains NaN values.")
                if np.isinf(img_array).any():
                    print("❌ Validation Error: Image contains Infinity values.")
            
            # --- 3. Content Check (Blank/Empty Image) ---
            # Check if the image is just a single solid color (e.g., all black or all white)
            extrema = img.getextrema()
            # For RGB, extrema returns ((minR, maxR), (minG, maxG), (minB, maxB))
            
            if all(low == high for low, high in (extrema if isinstance(extrema[0], tuple) else [extrema])):
                print("⚠️ Warning: Image is a solid color (blank).")
        
        # Return a single PIL if the original input wasn't a list, 
        # or the full list if it was.
        return pil_images if isinstance(input_data, list) else pil_images[0]

    # Process all potential inputs
    img1 = to_pil_list(data.get('image_1'), "RGB")
    img2 = to_pil_list(data.get('image_2'), "RGB")
    img1_obj_mask = to_pil_list(data.get('image_1_object_mask'), "L")
    img2_obj_mask = to_pil_list(data.get('image_2_object_mask'), "L")
    is_ssm = data.get('is_ssm', False)
    semantic_threshold = data.get('semantic_threshold', 0.7)

    score = mtgscorer(img1, img2, img1_obj_mask, img2_obj_mask, is_ssm=is_ssm,
                      semantic_threshold=semantic_threshold,)

    return {"score": score}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8101)
    args = parser.parse_args()
    host_ip = os.getenv("REWARD_NODE_IP", "0.0.0.0")
    print(f"Actually binding to: {host_ip}")
    print(f"✅ Starting Worker on internal port: {args.port}")
    uvicorn.run(app, host=host_ip, port=args.port)