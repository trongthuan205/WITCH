# WITCH: Reinforcement Learning with Multimodal Large Language Models for Verifiable and Temporally Grounded Video Captioning

WITCH (Weighted Interactive Temporal Captioning with Hint-based Rewards) generates video captions with a timestamp for each event claim. Its reinforcement learning stage uses pseudo-events extracted from the input video as supervision, without task-level human reference captions.

## Caption format

Each `<reference>` tag contains an event description and its estimated onset. The `mm:ss` timestamp records the first visible moment in whole seconds relative to the start of the clip.

```text
<reference time="00:02">A person places a bowl on the counter.</reference>
<reference time="00:06">They pour flour into the bowl.</reference>
```

## How it works

1. A captioning policy initialized from Qwen2.5-VL-7B-Instruct generates timestamped claims.
2. A separate, frozen instance of `Qwen/Qwen2.5-VL-7B-Instruct` generates a chronological list of visible events and their estimated onsets from sampled frames.
3. WITCH parses the tags and aligns each supported claim with a pseudo-event using text similarity. The alignment reward scores temporal order (TR), timestamp agreement (TC), and grounding coverage (GQ).
4. T-GRPO trains the policy using these rewards and a temporal bonus for qualifying rollouts. During inference, WITCH uses the reward to re-rank candidates and applies bounded prefix correction when all initial candidates score poorly.

## Setup and data

The pretrained checkpoint is `Qwen/Qwen2.5-VL-7B-Instruct`. Full WITCH inference uses a trained captioning checkpoint alongside the frozen pseudo-event extractor.


The paper evaluates MSVD, MSR-VTT, and VATEX. Prepare the video files and dataset splits for the experiment you want to run. Benchmark caption evaluation uses the datasets' reference captions. The temporal audit uses separately annotated event descriptions and onsets.

## Training

T-GRPO generates rollouts from chronological and temporally shuffled views of the same video. A qualifying rollout follows the tag protocol, has at least two supported claims whose matched pseudo-event indices strictly increase in caption order, and has a finite timestamp-agreement score at or above the threshold. When the ordered group's qualification rate meets or exceeds the shuffled group's rate, only qualifying ordered rollouts receive the bonus. Advantages are standardized across the full ordered group.

| Setting | Reported value |
| --- | --- |
| Training input | Up to 16 uniformly sampled frames per video |
| Per-frame visual budget | 128 × 28 × 28 |
| Ordered and shuffled rollouts per video | 8 and 4 |
| Maximum response length | 768 tokens |
| Optimizer | Adam |
| Learning rate | `1e-6` |
| Weight decay | `0.01` |
| Gradient norm limit | `5` |
| KL weight | `0.04` |
| Temporal bonus weight | `0.10` |
| Timestamp-agreement threshold for qualification | `0.50` |

Policy optimization updates the multimodal connector and language model with the vision encoder frozen. The KL reference policy starts from the same checkpoint as the captioning policy and stays frozen within each training domain. In the cross-domain experiments, the captioning policy adapts from MSVD to either MSR-VTT or VATEX. Both policies start the target-domain stage from the final MSVD captioning checkpoint.


## Inference

The reported setup uses 16–64 uniformly sampled frames and eight initial caption candidates per video. The policy and the extractor receive the same frames with a per-frame visual budget of 256 × 28 × 28. WITCH reuses one pseudo-event timeline across all candidates and selects the caption with the highest total reward. Conditional prefix correction uses a fixed revision budget.


## Evaluation

For benchmark-level TR, TC, and GQ, WITCH scores each selected caption against the same pseudo-event timeline used during inference. These scores measure agreement with that timeline. For the independent audit, the same captions are also scored against human timelines on a 100-video subset of the MSR-VTT test set. Among Full WITCH claims matched to human events, 76.2% are within 1 second of the annotated onset and the mean absolute onset error is 0.89 seconds (Table 8, Panel C).


### Benchmark results

Scores for Full WITCH as reported in Table 2 of the manuscript.

| Dataset | BLEU | METEOR | ROUGE-L | CIDEr |
| --- | ---: | ---: | ---: | ---: |
| MSVD | 68.2 | 43.8 | 89.4 | 134.5 |
| MSR-VTT | 61.2 | 37.5 | 75.4 | 106.2 |
| VATEX | 46.7 | 32.5 | 67.3 | 79.5 |

## Compute

The reported training run took approximately 8 hours for 1,000 updates on eight NVIDIA A100 GPUs. Inference was profiled in BF16 on one NVIDIA A100 (80 GB) with eight initial candidates per video.

| Inference setting | Time per video | Peak GPU memory |
| --- | ---: | ---: |
| Full WITCH | ~33 s | 55 GB |
| Generation only, using the same trained policy | ~20 s | 38 GB |
