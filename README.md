# moep

Prune MoE experts out of big open-weight models so they fit on machines people
actually own, then check honestly whether that was a good idea.

This is the pipeline behind the paper
*Half the Experts, All the Code* (arXiv link coming; study on Qwen3.6-35B-A3B
and Gemma-4-26B-A4B). The short version of what we found: removing half the
experts of a general MoE with coding-focused calibration is essentially free
for coding. Below that, don't prune - quantize. We tried five different ways
to make deep pruning beat 2-3 bit quantization at the same memory and lost
every time, which is itself the result.

## Layout

- `src/moep/` - the pipeline. One profiling pass collects per-expert routing
  and activation statistics; experts are ranked under interchangeable criteria
  (REAP, routing mass, activation norm, a domain-contrastive ratio we
  introduce and then argue against); surgery deletes experts shard-by-shard
  without ever loading the model and verifies surviving weights
  byte-for-byte; aggressive prunes can be healed by distilling from the base
  model with the surviving experts frozen; export produces GGUF with imatrix
  quantization.
- `tests/` - pytest suite against a tiny synthetic MoE that runs in seconds.
  The invariant that matters: pruning zero experts must reproduce base logits
  exactly.
- `artifacts/selections/` - the exact keep-set JSON for every pruned point in
  the paper, both families. If you want to reproduce a checkpoint, this
  plus the base model and `moep prune` is enough.

## Checkpoints

Pruned and quantized GGUFs:
[huggingface.co/anik-jha/Qwen3.6-35B-A3B-coding-reap50-GGUF](https://huggingface.co/anik-jha/Qwen3.6-35B-A3B-coding-reap50-GGUF). If you just want something to run: the 50%-keep Qwen specialist at
Q4_K_M or Q5_K_M with imatrix is the point of the whole exercise - at 11-13 GB
it beats 2-bit quantization of the full model on HumanEval+ at the same
memory.

## Calibration data

Not redistributed. The mix is 2,047 chat samples: Evol-CodeAlpaca, Hermes
function-calling, SWE-smith trajectories, and Tulu-3 as general ballast,
70/30 code to general, packed to 3.15M tokens. Section 4 of the paper has
the exact counts; all four sources are public.

Everything ran on a single DGX Spark (GB10, 119 GB unified memory). Nothing
here needs a cluster.

Code is MIT. Pruned Qwen checkpoints inherit Apache-2.0 from upstream; Gemma
derivatives follow the Gemma terms of use.
