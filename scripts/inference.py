import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import argparse
import os
from peft import PeftModel
import time

def main(args):
    pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
    pipe.to(args.device)
    
    if args.lora_path:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_path)
        pipe.transformer = pipe.transformer.merge_and_unload()
        print("Loaded LoRA weights from", args.lora_path)

    input_image = load_image(args.image_path)
    prompt = args.prompt
    base_folder = os.path.dirname(args.image_path)
    output_folder = os.path.join(args.output_folder, os.path.basename(base_folder))
    os.makedirs(output_folder, exist_ok=True)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(args.image_path))[0]}_seed_{args.seed}.png")
        
    image = pipe(
        image=input_image,
        prompt=prompt,
        guidance_scale=2.5,
        generator=generator
    ).images[0]
    
    image.save(output_path)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="path to the input image.")
    parser.add_argument("--prompt", type=str, required=True, help="the prompt to guide the generation.")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="path to the LoRA weights to load.")
    parser.add_argument("--is_use_mask", action='store_true',
                        help="whether to use masks to extract the reference object.")
    parser.add_argument("--output_folder", type=str, default="./output_folder",
                        help="folder to save the output images.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="device to run the model on.")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed for reproducibility.")
    args = parser.parse_args()
    main(args)