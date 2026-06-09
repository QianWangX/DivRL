import ml_collections
import importlib
import os

base = importlib.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    # prompting
    config.prompt_fn = "general_ocr"

    return config

def mtg_flux_kontext_syncd_1gpu():
    gpu_number=1
    config = compressibility()
    config.dataset_type = "subject_driven"
    config.dataset_name = "QWW/Syncd_filtered"
    config.dataset_total_samples=25
    config.train_as_val=True
    config.dataset_num_val = 3
    config.dataset_val_prompt_k = 3
    config.dataset_num_test = 1
    # Flux-Kontext
    config.pretrained.model = "black-forest-labs/FLUX.1-Kontext-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 2.5

    config.resolution = 512
    config.sample.train_batch_size = 21
    config.sample.micro_batch_size = 3
    config.sample.num_processes = gpu_number
    config.sample.num_image_per_prompt = 21
    mbs = config.sample.get('micro_batch_size', config.sample.train_batch_size)
    config.sample.num_batches_per_epoch = int(2/(gpu_number*mbs/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 2 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.noise_level = 0.9
    config.mixed_precision = "bf16"
    config.save_freq = 10 # epoch
    config.eval_freq = 1
    config.save_dir = 'logs/mtg_syncd_7gpu/flux_kontext'
    config.reward_fn = {
        "mtgscore_vsm": 1,
        "mtgscore_nssm": 1,
    }
    config.is_spatial = False
    config.is_hinge = True
    config.hinge_dict = {
        "hinge_main": "mtgscore_nssm",
        "hinge_gate": "mtgscore_vsm",
        "hinge_margin": 0.5,
        "hinge_weight": 5,
    }
    config.spatial_norm = "spatial"
    config.per_prompt_stat_tracking = False
    return config


def divrl_flux_kontext_syncd_7gpu_stage_1():
    gpu_number=7
    config = compressibility()
    config.dataset_type = "subject_driven"
    config.dataset_name = "QWW/Syncd_filtered"
    config.dataset_num_val = 50
    config.dataset_val_prompt_k = 3
    config.dataset_num_test = 500
    # Flux-Kontext
    config.pretrained.model = "black-forest-labs/FLUX.1-Kontext-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 2.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.micro_batch_size = 3
    config.sample.num_processes = gpu_number
    config.sample.num_image_per_prompt = 21
    mbs = config.sample.get('micro_batch_size', config.sample.train_batch_size)
    config.sample.num_batches_per_epoch = int(32/(gpu_number*mbs/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 2 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.num_epochs = 100
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.noise_level = 0.9
    config.mixed_precision = "bf16"
    config.save_freq = 5 # epoch
    config.eval_freq = 5
    config.save_dir = 'logs/divrl_syncd_7gpu/flux_kontext'
    config.reward_fn = {
        "mtgscore_nssm": 1,
    }
    config.is_hinge = False
    config.hinge_dict = None
    config.per_prompt_stat_tracking = False
    return config


def divrl_flux_kontext_syncd_7gpu_stage_2():
    gpu_number=7
    config = compressibility()
    config.dataset_type = "subject_driven"
    config.dataset_name = "QWW/Syncd_filtered"
    config.dataset_num_val = 50
    config.dataset_val_prompt_k = 3
    config.dataset_num_test = 500
    # Flux-Kontext
    config.pretrained.model = "black-forest-labs/FLUX.1-Kontext-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 2.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.micro_batch_size = 3
    config.sample.num_processes = gpu_number
    config.sample.num_image_per_prompt = 21
    mbs = config.sample.get('micro_batch_size', config.sample.train_batch_size)
    config.sample.num_batches_per_epoch = int(32/(gpu_number*mbs/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 2 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.num_epochs = 100
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.noise_level = 0.9
    config.mixed_precision = "bf16"
    config.save_freq = 5 # epoch
    config.eval_freq = 5
    config.save_dir = 'logs/divrl_syncd_7gpu/flux_kontext'
    config.reward_fn = {
        "mtgscore_vsm": 1,
        "mtgscore_nssm": 1,
    }
    config.is_hinge = True
    config.hinge_dict = {
        "hinge_main": "mtgscore_nssm",
        "hinge_gate": "mtgscore_vsm",
        "hinge_margin": 0.5,
        "hinge_weight":5,
    }
    config.per_prompt_stat_tracking = False
    return config
