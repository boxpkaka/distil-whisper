# Distil-Whisper

[[Paper]](https://arxiv.org/abs/2311.00430)
[[Models]](https://huggingface.co/collections/distil-whisper/distil-whisper-models-65411987e6727569748d2eb6)
[[wandb]](https://wandb.ai/sanchit-gandhi/distil-whisper/workspace?workspace=user-sanchit-gandhi)

Distil-Whisper is a distilled version of Whisper that is **6 times faster**, 49% smaller, and performs **within 1% WER** on 
out-of-distribution evaluation sets.

| Model              | Link                            |
|--------------------|---------------------------------|
| `distil-medium.en` | To be published on November 2nd |
| `distil-large-v2`  | To be published on November 2nd |

## 1. Usage 👨‍💻

The Distil-Whisper checkpoints will be released on November 2nd with a direct 🤗 Transformers integration. Instructions 
for running inference will be provided here:

```python
from transformers import WhisperForConditionalGeneration

...
```

## 2. Why use Distil-Whisper? ⁉️

Distil-Whisper is designed to be a drop-in replacement for Whisper on English speech recognition. Here are 5 reasons for making the
switch to Distil-Whisper:

1. **Faster inference:** 6 times faster inference speed, while performing to within 1% WER of Whisper on out-of-distribution audio:

<p align="center">
  <img src="https://huggingface.co/datasets/distil-whisper/figures/resolve/main/main_table.png?raw=true" width="600"/>
</p>

2. **Robustness to noise:** demonstrated by strong WER performance at low signal-to-noise ratios:

<p align="center">
  <img src="https://huggingface.co/datasets/distil-whisper/figures/resolve/main/noise.png?raw=true" width="600"/>
</p>

3. **Robustness to hallucinations:** quantified by 1.3 times fewer repeated 5-gram word duplicates (5-Dup.) and 2.1% lower insertion error rate (IER) than Whisper:

<p align="center">
  <img src="https://huggingface.co/datasets/distil-whisper/figures/resolve/main/hallucination.png?raw=true" width="600"/>
</p>

4. **Designed for speculative decoding:** Distil-Whisper can be used as an assistant model to Whisper, giving 2 times faster inference speed while mathematically ensuring the same outputs as the Whisper model.
5. **Permissive license:** Distil-Whisper is [MIT licensed](./LICENSE), meaning it can be used for commercial applications.

## 3. Approach ✍️

To distill Whisper, we copy the entire encoder module and freeze it during training. We copy only two decoder layers, 
which are initialised from the first and last decoder layers from Whisper. All other decoder layers from Whisper
are discarded:

<p align="center">
  <img src="https://huggingface.co/datasets/distil-whisper/figures/resolve/main/architecture.png?raw=true" width="600"/>
</p>

Distil-Whisper is trained on a *knowledge distillation* objective. Specifically, it is trained to minimise the KL divergence
between the distilled model and the Whisper model, as well as the cross-entropy loss on pseudo-labelled audio data.

We train Distil-Whisper on a total of 22k hours of pseudo-labelled audio data, spanning 10 domains with over 18k speakers:

<p align="center">
  <img src="https://huggingface.co/datasets/distil-whisper/figures/resolve/main/datasets.png?raw=true" width="600"/>
</p>

This diverse audio dataset is paramount to ensuring robustness of Distil-Whisper to different datasets and domains. 

In addition, we use a WER filter to discard pseudo-labels where Whisper mis-transcribes or hallucinates. This greatly 
improves WER performance of the downstream distilled model.

For full details on the distillation set-up and evaluation results, refer to the [Distil-Whisper paper](https://arxiv.org/abs/2311.00430).

## 4. Training Code

Training code to reproduce Distil-Whisper will be published here shortly. We will also release more general code to distill
Whisper for multilingual speech recognition, facilitating anyone in the community to distill Whisper on their choice of 
language.

## 5. Acknowledgements
* OpenAI for the Whisper [model](https://huggingface.co/openai/whisper-large-v2) and [original codebase](https://github.com/openai/whisper)
* Hugging Face 🤗 [Transformers](https://github.com/huggingface/transformers) for the model integration
* Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) programme for Cloud TPU v4s

## 6. Citation

If you use this model, please consider citing the Distil-Whisper paper:
```
@misc{gandhi2023distilwhisper,
      title={Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling}, 
      author={Sanchit Gandhi and Patrick von Platen and Alexander M. Rush},
      year={2023},
      eprint={2311.00430},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

And also the Whisper paper:
```
@misc{radford2022robust,
      title={Robust Speech Recognition via Large-Scale Weak Supervision}, 
      author={Alec Radford and Jong Wook Kim and Tao Xu and Greg Brockman and Christine McLeavey and Ilya Sutskever},
      year={2022},
      eprint={2212.04356},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
