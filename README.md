# RL-LLM-NLP
This repository encompasses libraries and papers on Reinforcement Learning (RL) within Large Language Models (LLM) and Natural Language Processing (NLP).

I consider RL to be a pivotal technology in the field of AI, and NLP (particularly LLM) to be a direction well worth exploring.

## Library

| GitHub                                                       | From        | Year | Desc                                                         |
| ------------------------------------------------------------ | ----------- | ---- | ------------------------------------------------------------ |
| [PRIME](https://github.com/PRIME-RL/PRIME)                   | PRIME-RL    | 2025 | Scalable RL solution for the advanced reasoning of language models |
| [rStar](https://github.com/microsoft/rStar)                  | MicroSoft   | 2025 |                                                              |
| [veRL](https://github.com/volcengine/verl)                   | Bytedance   | 2024 | Volcano Engine Reinforcement Learning for LLM                |
| [trl](https://github.com/huggingface/trl)                    | HuggingFace | 2024 | Train LM with RL                                             |
| [RL4LMs](https://github.com/allenai/RL4LMs)                  | Allen       | 2023 | RL library to fine-tune LM to human preferences              |
| [alignment-handbook](https://github.com/huggingface/alignment-handbook) | huggingface | 2023 | Robust recipes to align language models with human and AI preferences |

## Paper

| Cate       | Abbr                   | Title                                                        | From                    | Year | Link                                                         |
| ---------- | ---------------------- | ------------------------------------------------------------ | ----------------------- | ---- | ------------------------------------------------------------ |
| RL         | MRT                    | Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning | Carnegie Mellon         | 2025 | [paper](http://arxiv.org/abs/2503.07572), [GitHub](https://github.com/CMU-AIRe/MRT) |
| RL         | L1, LCPO               | L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning | Carnegie Mellon         | 2025 | [paper](http://arxiv.org/abs/2503.04697), [GitHub](https://github.com/cmu-l3/l1) |
| RL         | Online-DPO-R1          | Online-DPO-R1: Unlocking Effective Reasoning Without the PPO Overhead | Salesforce AI Research  | 2025 | [paper](https://efficient-unicorn-451.notion.site/Online-DPO-R1-Unlocking-Effective-Reasoning-Without-the-PPO-Overhead-1908b9a70e7b80c3bc83f4cf04b2f175), [GitHub](https://github.com/RLHFlow/Online-DPO-R1) |
| RL         | orz                    | Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model | StepFun                 | 2025 | [paper](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/ORZ_paper.pdf), [GitHub](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main) |
| RL         | OREAL                  | Exploring the Limit of Outcome Reward for Learning Mathematical Reasoning | InternLM                | 2025 | [paper](https://arxiv.org/abs/2502.06781), [GitHub](https://github.com/InternLM/OREAL) |
| RL         | R1                     | DeepSeek-R1                                                  | DeepSeek                | 2025 | [paper](https://github.com/deepseek-ai/DeepSeek-R1), ①       |
|            |                        |                                                              |                         |      |                                                              |
| o1         | Sky-T1                 | Sky-T1: Train your own O1 preview model within $450          | NovaSky-AI              | 2025 | [GitHub](https://github.com/NovaSky-AI/SkyThought)           |
| o1         | STILL                  | A series of technical report on Slow Thinking with LLM       | RUCAIBox                | 2025 | [GitHub](https://github.com/RUCAIBox/Slow_Thinking_with_LLMs) |
|            |                        |                                                              |                         |      |                                                              |
| RL Scaling | LIMR                   | LIMR: Less is More for RL Scaling                            | GAIR-NLP                | 2025 | [paper](https://arxiv.org/abs/2502.11886), [GitHub](https://github.com/GAIR-NLP/LIMR) |
| RL Scaling | DeepScaleR             | DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL | Agentica                | 2025 | [paper](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2), [GitHub](https://github.com/agentica-project/deepscaler) |
| RL Scaling | ScalingLaw             | Value-Based Deep RL Scales Predictably                       | Berkeley                | 2025 | [paper](https://arxiv.org/abs/2502.04327)                    |
|            |                        |                                                              |                         |      |                                                              |
| SLM        | PRIME                  | Process Reinforcement through Implicit Rewards               | PRIME-RL                | 2025 | [paper](https://curvy-check-498.notion.site/Process-Reinforcement-through-Implicit-Rewards-15f4fcb9c42180f1b498cc9b2eaf896f), [GitHub](https://github.com/PRIME-RL/PRIME) |
| SLM        | rStar-Math             | rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking | MicroSoft               | 2025 | [paper](https://arxiv.org/abs/2501.04519), [GitHub](https://github.com/microsoft/rStar) |
| SLM        | rStar                  | rStar: Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers | MicroSoft               | 2024 | [paper](https://arxiv.org/pdf/2408.06195), [GitHub](https://github.com/zhentingqi/rStar) |
|            |                        |                                                              |                         |      |                                                              |
| Unlearn    |                        | A Closer Look at Machine Unlearning for Large Language Models | Sea AI                  | 2024 | [paper](https://arxiv.org/abs/2410.08109v1), [GitHub](https://github.com/sail-sg/closer-look-LLM-unlearning) |
| Unlearn    | Quark                  | Quark: Controllable Text Generation with Reinforced [Un]learning | Allen                   | 2022 | [paper](http://arxiv.org/abs/2205.13636), [GitHub](https://github.com/GXimingLu/Quark) |
|            |                        |                                                              |                         |      |                                                              |
| Align      | ReMax                  | ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models | CUHK                    | 2024 | [paper](https://arxiv.org/abs/2310.10505)                    |
| Align      |                        | A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and More | Salesforce              | 2024 | [paper](https://arxiv.org/abs/2407.16216)                    |
| Align      |                        | Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback | Allen                   | 2024 | [paper](https://arxiv.org/abs/2406.09279), [GitHub](https://github.com/hamishivi/EasyLM) |
| Align      |                        | Preference Tuning with Human Feedback on Language, Speech, and Vision Tasks: A Survey | Capital One             | 2024 | [paper](http://arxiv.org/abs/2409.11564)                     |
| Align      | RLHF                   | Training language models to follow instructions with human feedback | OpenAI                  | 2022 | [paper](https://arxiv.org/abs/2203.02155)                    |
| Align      | NLPO                   | Is Reinforcement Learning (Not) for Natural Language Processing: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization | Allen                   | 2022 | [paper](http://arxiv.org/abs/2210.01241), [GitHub](https://github.com/allenai/rl4lms) |
| Align      |                        | Fine-Tuning Language Models from Human Preferences           | OpenAI                  | 2020 | [paper](http://arxiv.org/abs/1909.08593), [GitHub](https://github.com/openai/lm-human-preferences) |
| Align      | RLOO                   | Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs | Cohere                  | 2024 | [paper](https://arxiv.org/abs/2402.14740)                    |
|            |                        |                                                              |                         |      |                                                              |
| Policy     | Dr. DAPO               | Understanding R1-Zero-Like Training: A Critical Perspective  | Sea AI Lab              | 2025 | [paper](http://arxiv.org/abs/2503.20783), [GitHub](https://github.com/sail-sg/understand-r1-zero) |
| Policy     | DAPO                   | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | ByteDance Seed          | 2025 | [paper](https://arxiv.org/abs/2503.14476), [GitHub](https://github.com/BytedTsinghua-SIA/DAPO) |
| Policy     | GRPO                   | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | DeepSeek                | 2024 | [paper](https://arxiv.org/abs/2402.03300)                    |
| Policy     | DPO                    | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | Stanford                | 2024 | [paper](https://arxiv.org/abs/2305.18290)                    |
| Policy     |                        | Decision Transformer: Reinforcement Learning via Sequence Modeling | Berkeley                | 2021 | [paper](https://arxiv.org/abs/2106.01345), [GitHub](https://github.com/kzl/decision-transformer) |
| Policy     | PPO                    | Proximal Policy Optimization Algorithms                      | OpenAI                  | 2017 | [paper](https://arxiv.org/abs/1707.06347)                    |
| Policy     | REINFORCE multi-sample | Buy 4 Reinforce Samples, Get a Baseline for Free!            | University of Amsterdam | 2019 | [paper](https://openreview.net/pdf?id=r1lgTGL5DE)            |
| Policy     | REINFORCE              | Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning | Northeastern University | 1992 | [paper](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) |

## Appendix

- ① DeepSeek-R1相关复现：
    - [Jiayi-Pan/TinyZero: Clean, minimal, accessible reproduction of DeepSeek R1-Zero](https://github.com/Jiayi-Pan/TinyZero)
    - [huggingface/open-r1: Fully open reproduction of DeepSeek-R1](https://github.com/huggingface/open-r1)
    - [hkust-nlp/simpleRL-reason: This is a replicate of DeepSeek-R1-Zero and DeepSeek-R1 training on small models with limited data](https://github.com/hkust-nlp/simpleRL-reason)
    - [ZihanWang314/RAGEN: RAGEN is the first open-source reproduction of DeepSeek-R1 on AGENT training.](https://github.com/ZihanWang314/ragen)
