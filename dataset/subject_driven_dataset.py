import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from datasets import load_dataset


class SubjectDrivenDataset(Dataset):
    def __init__(self, dataset_name="QWW/Syncd_filtered", split='train', base_seed=0,
                 total_samples=None, train_as_val=False, num_val=None, percent_val=0.95, num_test=None,
                 val_prompt_k=3, resolution=512):

        self.base_seed = base_seed
        self.resolution = resolution

        # Load full train split — all splitting is handled below
        hf_dataset = load_dataset(dataset_name, split="train")
        indices = list(range(len(hf_dataset)))

        random.seed(base_seed)
        random.shuffle(indices)

        if total_samples is not None:
            indices = indices[:total_samples]

        if num_test is not None and num_test < len(indices):
            test_indices = indices[-num_test:]
            indices = indices[:-num_test]
        else:
            test_indices = []
            print("Warning: num_test is not specified or exceeds available samples, using all data for training/validation")

        self.split = split

        if split == "train" or (split == "val" and train_as_val):
            if num_val is not None:
                indices = indices[:-num_val]
            else:
                indices = indices[:int(len(indices) * percent_val)]
            print(f"Number of training samples: {len(indices)}")
        elif split == "val":
            if num_val is not None:
                indices = indices[-num_val:]
            else:
                indices = indices[int(len(indices) * percent_val):]
            print(f"Number of validation samples: {len(indices)}")
        elif split == "test":
            assert num_test is not None, "num_test must be specified for test split"
            indices = test_indices
            print(f"Number of test samples: {len(indices)}")
        else:
            raise ValueError("Unsupported split")

        self.indices = indices
        self.hf_dataset = hf_dataset

        # Build (img_idx, prompt_idx) pairs for val/test
        self.entries = []
        if self.split != 'train':
            for img_idx, ds_idx in enumerate(self.indices):
                captions = self._parse_captions(hf_dataset[ds_idx]['caption'])
                prompts_to_use = len(captions) if split == 'test' else min(len(captions), val_prompt_k)
                for p_idx in range(prompts_to_use):
                    self.entries.append((img_idx, p_idx))

        self.epoch = 0

    def _parse_captions(self, caption_str):
        """Parse the caption column, which may be a multi-line txt file or a plain string.

        Original txt format: first line is a header (skipped), remaining lines are
        individual caption variations. If only one line exists (plain caption), use it
        directly so the dataset still works when the txt structure is absent.
        """
        if not caption_str:
            return [""]
        lines = [l.strip() for l in caption_str.split('\n') if l.strip()]
        if len(lines) > 1:
            return lines[1:]   # skip header line, matching original txt loading
        elif len(lines) == 1:
            return lines       # plain single caption — no header to skip
        return [""]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        if self.split == "train":
            return len(self.indices)
        return len(self.entries)

    def __getitem__(self, idx):
        g = torch.Generator(device='cpu')
        if self.split == 'train':
            g.manual_seed(self.base_seed + idx + self.epoch)
            img_idx = idx
            ds_idx = self.indices[img_idx]
            captions = self._parse_captions(self.hf_dataset[ds_idx]['caption'])
            prompt_idx = torch.randint(0, len(captions), (1,), generator=g).item()
        else:
            img_idx, prompt_idx = self.entries[idx]
            ds_idx = self.indices[img_idx]
            captions = self._parse_captions(self.hf_dataset[ds_idx]['caption'])

        item = self.hf_dataset[ds_idx]

        image = item['image'].convert("RGB")
        image = image.resize((self.resolution, self.resolution), resample=Image.LANCZOS)
        mask = item['mask'].convert("L")
        mask = mask.resize((self.resolution, self.resolution), resample=Image.LANCZOS)

        prompt = captions[min(prompt_idx, len(captions) - 1)]

        # Use id/file_name column as image name if available, fall back to dataset index
        image_name = item.get('file_name', item.get('id', str(ds_idx)))
        if not isinstance(image_name, str):
            image_name = str(image_name)

        return {
            "ref_image": image,
            "ref_image_mask": mask,
            "prompt": prompt,
            "prompt_with_image_path": prompt,
            "metadata": {
                "img_name": image_name,
                "img_idx": img_idx,
                "prompt_idx": prompt_idx,
            }
        }

    @staticmethod
    def collate_fn(examples):
        ref_images = [example["ref_image"] for example in examples]
        ref_image_masks = [example["ref_image_mask"] for example in examples]
        prompts = [example["prompt"] for example in examples]
        if "prompt_bg" in examples[0]:
            prompts_bg = [example["prompt_bg"] for example in examples]
        else:
            prompts_bg = None
        metadatas = [example["metadata"] for example in examples]
        prompt_with_image_paths = [example["prompt_with_image_path"] for example in examples]

        if prompts_bg:
            return prompts, prompts_bg, metadatas, prompt_with_image_paths, ref_images, ref_image_masks
        else:
            return prompts, metadatas, prompt_with_image_paths, ref_images, ref_image_masks
