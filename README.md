# Fine-tuning Qwen2.5-VL 3B

This repository contains a script for training [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) and [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) with only using HuggingFace and [Liger-Kernel](https://github.com/linkedin/Liger-Kernel).


## Supported Features

- Deepspeed
- LoRA/QLoRA
- Full-finetuning
- Enable finetuning `vision_model` while using LoRA.
- Disable/enable Flash Attention 2
- Multi-image and video training
- Training optimized with liger kernel
- Mixed-modality dataset
- Direct Preference Optimization (DPO)
- Group Relative Policy Optimization (GRPO)


## Installation

### 1.Environments

- Ubuntu 22.04
- Nvidia-Driver 570
- Cuda version 12.8

Install the required packages using `environment.yaml`.

### 2.Using `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate train
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```
### 3.Download Qwen2.5-VL-3B
It is recommended to use [ModelScope](https://modelscope.cn/models/qwen/Qwen2.5-VL-3B-Instruct/) for model downloading.

```bash
pip install modelscope
modelscope download --model qwen/Qwen2.5-VL-3B-Instruct 
```

## Dataset

The script requires a dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.<br>

### 1.VQA dataset

Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

### 2.R2R&RXR dataset

We provide annotations for`r2r`, `rxr`on [Hugging Face](https://huggingface.co/datasets/a8cheng/NaVILA-Dataset).
Please download the repo and extract the `tar.gz` files in their respective subfolders. 

The data should have structure like:

```graphql
├─ R2R
|   ├─ train
|   |    ├─ 1
|   |    |    ├─ frame_0.jpg 
|   |    |    ├─ ...
|   |    ├─ ...
|   ├─ annotations.json
├─ RxR
|   ├─ train
|   |    ├─ 1
|   |    |    ├─ frame_0.jpg 
|   |    |    ├─ ...
|   |    ├─ ...
|   ├─ annotations.json
```
convert to llava format as follow

```json
[
  {
    "id": "000000033471",
    "image": ["000000033471.jpg", "000000033472.jpg"],
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n<image>\nIs the perspective of the camera differnt?"
      },
      {
        "from": "gpt",
        "value": "Yes, It the perspective of the camera is different."
      }
    ]
  }
  ...
]
```

## Supervised Fine Tuning

**Note:** Deepspeed zero2 is faster than zero3, however it consumes more memory. Also, most of the time zero2 is more stable than zero3.<br><br>
**Tip:** You could use `adamw_bnb_8bit` for optimizer to save memory.

To run the training script, use the following command:

### Full Finetuning

```bash
bash scripts/finetune.sh
```

### Finetune with LoRA

**Note:** Liger-kernel won't work with QLoRA. You need to disable to use QLoRA.<br>
If you want to train only the language model with LoRA and perform full training for the vision model:

```bash
bash scripts/finetune_lora.sh
```

If you want to train both the language model and the vision model with LoRA:

```bash
bash scripts/finetune_lora_vision.sh
```

**IMPORTANT:** If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together.

<details>
<summary>Training arguments</summary>

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the Qwen2-VL model. **(Required)**
- `--use_liger` (bool): Option for using liger kernel to save memory.
- `--output_dir` (str): Output directory for model checkpoints
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze vision_model (default: False).
- `--freeze_llm` (bool): Option to freeze LLM (default: False).
- `--freeze_merger` (bool): Option to tune projector (default: False).
- `--num_lora_modules` (int): Number of target modules to add LoRA (-1 means all layers).
- `--vision_lr` (float): Learning rate for vision_model.
- `--merger_lr` (float): Learning rate for merger(projector).
- `--learning_rate` (float): Learning rate for language module.
- `--bf16` (bool): Option for using bfloat16.
- `--fp16` (bool): Option for using fp16.
- `--image_min_pixels` (int): Option for minimum input tokens for image.
- `--image_max_pixles` (int): Option for maximum maxmimum tokens for image.
- `--video_min_pixels` (int): Option for minimum input tokens for video.
- `--video_max_pixles` (int): Option for maximum maxmimum tokens for video.
- `--image_resized_width` (int): Option for setting the width of the input image.
- `--image_resized_height` (int): Option for setting the height of the input image.
- `--video_resized_width` (int): Option for setting the width of the input video.
- `--video_resized_height` (int): Option for setting the height of the input video.
- `--fps` (float): Frames per second for video data.
- `--nframes` (int): Number of frames for video data.
- `--lora_enable` (bool): Option for using LoRA.
- `--vision_lora` (bool): Option for including `vision_tower` in LoRA module. `lora_enable` should be `True` to use this option.
- `--use_dora` (bool): Option for using DoRA instead of LoRA. `lora_enable` should be `True` to use this option.
- `--lora_namespan_exclude` (str): Exclude modules with namespans to add LoRA.
- `--max_seq_length` (int): Maximum sequence length (default: 32K).
- `--bits` (int): Quantization bits (default: 16).
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--lora_rank` (int): LoRA rank (default: 128).
- `--lora_alpha` (int): LoRA alpha (default: 256).
- `--lora_dropout` (float): LoRA dropout (default: 0.05).
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).

**Note:** The learning rate of `vision_model` should be 10x ~ 5x smaller than the `language_model`.

</details>

### Train with video dataset

You can train the model using a video dataset. You can set LoRA configs and use for LoRA too.<br>
**Note:** You could not set `fps` and `nframes` at the same time.

```bash
bash scripts/finetune_video.sh
```

**Note:** When training with video, it just as multi-image so you should adjust the `max_pixels` for maximum resolution and `fps` based on the available VRAM.

If you run out of vram, you can use [zero3_offload](./scripts/zero3_offload.json) instead of [zero3](./scripts/zero3_offload.json).<br>
You could use [zero2_offload](./scripts/zero2_offload.json) for a bit faster training.

#### Image Resolution for vram usage

The model supprots a wide range of resolution inputs. By default, it uses the native resolution for input.
For better performance using native or higer pixel numbers are recommended, however it takes too much memory and computation time for large images. So you could adjust the pixel numbers for it.
The model splits the image into `token * 28 * 28` so you could just change the the token_num part in the script. <br>
For example:

```
--image_min_pixels $((256 * 28 * 28))
--image_max_pixels $((1280 * 28 * 28))
--video_min_pixels $((128 * 28 * 28))
--video_max_pixels $((768 * 28 * 28))
```

Besides you could directly set the image/video height and width to control over the memory.

```
--image_resized_width 448
--image_resized_height 448
--video_resized_width 448
--video_resized_height 448
```

These values will be rounded to the nearest multiple of 28.

#### Merge LoRA Weights

```
bash scripts/merge_lora.sh
```

**Note:** Remember to replace the paths in `finetune.sh` or `finetune_lora.sh` with your specific paths. (Also in `merge_lora.sh` when using LoRA.)

## DPO Finetuning

You can train the model using Direct Preference Optimization (DPO).<br>
The process is quite similar to Supervised Fine-Tuning (SFT), and you can also apply LoRA during DPO training just like in SFT.

```bash
bash scripts/finetune_dpo.sh
```

Most of the training arugments are same as SFT, but few other arguments are added for DPO training.

<details>
<summary>Training arguments</summary>

- `--dpo_loss` (str): Loss type for dpo. (default: 'sigmoid')
- `--precompute_ref_log_probs` (bool): Wheter to precompute the reference log probs (default: False)
- `--beta` (float): The beta value for DPO (default: 0.1)

</details>

## GRPO Finetuning

You can traing the model using Group Relative Policy Optimization (GRPO) <br>
The process is quite similar to Supervised Fine-Tuning (SFT), and you can also apply LoRA during GRPO training just like in SFT.<br>
<br>

### Prerequisites

| What                      | Where                       | Notes                                                                                       |
| ------------------------- | --------------------------- | ------------------------------------------------------------------------------------------- |
| **Reward functions**      | `src/train/reward_funcs.py` | Add any function that ends with `_reward`. The training script picks them up automatically. |
| **Custom system prompts** | `src/constants.py`          | Append your own prompt strings here.                                                        |

You could start training using this script.<br>
Before training, **Please check the dataset format once more.** The format is a bit different from other training methods.

```bash
bash scripts/finetune_grpo.sh
```

Most of the training arugments are same as SFT, but few other arguments are added for GRPO training.

<details>
<summary>Training arguments</summary>

- `--temperature` (float): Generation config (default: 0.9)
- `--top_p` (float): Generation config (default: 1.0)
- `--top_k` (int): Generation config (default: 50)
- `--min_p` (float): Generation config (default: None)
- `--repetition_penalty` (float): Generation config (default: 1.0)
- `--max_completion_length` (int): Max length for the completion (default: 256)
- `--max_prompt_length` (int): Max length for the prompt (default: 512)
- `--beta` (float): KL Coefficient. (default: 0.04)

</details>

**Note:** **Liger GRPO loss** and **vLLM back-end** are not yet supported. Both will be added soon.

## Classification Finetuning

### ⚠️This is an experimental feature.

The [model](src/model/modeling_cls.py) is tailored for classification tasks, such as other SequenceClassification models.

For the classification task, you need to prepare the dataset in a specific format. The dataset should be a JSON file where each entry contains an image and its corresponding label. The labels should be integers starting from 0.<br>
You can set the text in the filed `prompt` to provide a questions and options for the classification task. Also if your dataset dose not contain the `prompt` field, the script will automatically use the `USER_MESSAGE` from the [cls_dataset.py](src/dataset/cls_dataset.py).<br>

**Please see the example below for the dataset format.**<br>

<details>
<summary>Example for Classification Dataset</summary>

```json
[
  {
    "id": "06bc8a17-bb1c-4007-8c08-92c41e2628b2",
    "image": "image_2.jpg",
    "prompt": "Question: What is in the image? \n Options: \n 1. A train \n 2. A bus \n 3. A car \n 4. A bicycle",
    "label": "3",
  }
  ...
]
```

**Note:** You should set the `CLASS_2_ID` variable in the [cls_dataset.py](src/dataset/cls_dataset.py).

</details>

<br>

The dataset can contain **single/multi-image or video data**, and the model will be trained to classify the images/videos based on the provided labels.<br>

For now, you can select loss from one of the following:

- `cross_entropy`
- `focal_loss`
- `class_balanced_cross_entropy`
- `class_balanced_focal_loss`

Also you can set early stopping patience and threshold for the training.
For example, you can set `--early_stopping_patience 5` and `--early_stopping_threshold 0.01` to stop the training if the validation loss does not improve for 5 epochs with a threshold of 0.01.

Most of the training arugments are same as SFT, but few other arguments are added for classification training.

<details>
<summary>Training arguments</summary>

- `--loss_type` (str): Loss type for classification (default: 'cross_entropy').
- `--focal_alpha` (str): Focal Loss alpha value. If None use CrossEntropyLoss. ex '1.0,7.5' (default: None).
- `--focal_gamma` (float): Focal Loss gamma value. (default: 0.0)
- `--num_labels` (int): Number of labels for classification
- `--class_balanced_beta` (float): Class Balanced beta value. (default: 0.999)
- `--early_stopping_patience` (int): Early stopping patience (default: 0)
- `--early_stopping_threshold` (float): Early stopping threshold (default: 0.01)

</details>

You can run the training script using the following command:

```bash
bash scripts/finetune_cls.sh
```

#### Experimental Features

- Sampler for the dataet. The trainer scripts supports the sampler for the dataset. You could make your own sampler.

## Inference

**Note:** You should use the merged weight when trained with LoRA.

### Gradio Infernce (WebUI)

1. Install gradio

```
pip install gradio
```

2. Launch app

```
python -m src.serve.app \
    --model-path /path/to/merged/weight
```

You can launch gradio based demo with this command. This can also set some other generation configs like `repetition_penalty`, `temperature` etc.

## Issue for libcudnn error

```
Could not load library libcudnn_cnn_train.so.8. Error: /usr/local/cuda-12.1/lib/libcudnn_cnn_train.so.8: undefined symbol: _ZN5cudnn3cnn34layerNormFwd_execute_internal_implERKNS_7backend11VariantPackEP11CUstream_stRNS0_18LayerNormFwdParamsERKNS1_20NormForwardOperationEmb, version libcudnn_cnn_infer.so.8
```

You could run `unset LD_LIBRARY_PATH` for this error.
You could see this [issue](https://github.com/andimarafioti/florence2-finetuning/issues/2)

## TODO

- [x] Support for video data
- [x] Add demo for multi-image and video
- [x] Handle mixed-modality data in dataset and collator
- [x] Support Qwen2.5-VL
- [x] Monkey-patch liger-kernel for Qwen2.5-VL
- [x] Update the code base to the latest transformers.
- [x] Add DPO
- [x] Add GRPO
- [ ] Fix GRPO liger loss to work

## Known Issues

- [libcudnn issue](#issue-for-libcudnn-error)

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this repository useful in your project, please consider giving a :star: and citing:

```bibtex
@misc{Qwen2-VL-Finetuning,
  author = {Yuwon Lee},
  title = {Qwen2-VL-Finetune},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/2U1/Qwen2-VL-Finetune}
}
```

## Acknowledgement

This project is based on

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): An amazing open-source project of LMM.
- [Mipha](https://github.com/zhuyiche/llava-phi): Open-source projcet of SMM with amazing capabilites.
- [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct): Awesome pretrained MLLM based on Qwen2.
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel): Collection of Tirton kernels designed specifically for LLM training.
- [VLM-R1](https://github.com/om-ai-lab/VLM-R1): Open-source project of Reinforcement Learning with VLMs.




### Visual Instruction Tuning
