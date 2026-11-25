"""Step 2. Run one model over prompts.jsonl with vLLM, saving as it goes.

    python run.py --model gemma --limit 40      # smoke test
    python run.py --model gemma                 # full run (resumes automatically)
    python run.py --model qwen

Output: outputs_<model>.jsonl (one response per line) and run_meta_<model>.json
If the run stops, run the same command again: prompts already in the output file are skipped.

Settings: no system prompt, thinking off for both models.
  Gemma 4: thinking is off by default (no <|think|> token); sampling = its generation_config.json defaults
           (model card: temperature 1.0, top_p 0.95, top_k 64).
  Qwen3.5: thinking is on by default, so it is turned off with enable_thinking=False, and sampling = the
           model card's settings for non-thinking mode (temperature 0.7, top_p 0.8, top_k 20,
           presence_penalty 1.5). Add --qwen-thinking to keep Qwen's default thinking instead.
Only the seed and max_tokens are set per request.
"""
import argparse
import json
import os
import platform
import time

MODELS = {
    "gemma": dict(repo="google/gemma-4-12B-it", quantization="fp8", dtype="auto",
                  max_tokens=4096, mm_off={"image": 0, "audio": 0},
                  chat_template_kwargs=None, sampling=None),  # None = generation_config.json defaults
    "qwen": dict(repo="Qwen/Qwen3.5-9B", quantization=None, dtype="bfloat16",
                 max_tokens=4096, mm_off={"image": 0, "video": 0},
                 chat_template_kwargs={"enable_thinking": False},
                 sampling=dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
                               presence_penalty=1.5, repetition_penalty=1.0)),
}
QWEN_THINKING = dict(max_tokens=12288, chat_template_kwargs=None,  # template default = thinking on
                     sampling=dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                                   presence_penalty=1.5, repetition_penalty=1.0))


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["prompt_id"])
                except (json.JSONDecodeError, KeyError):
                    pass  # a half-written last line from a crash; that prompt is rerun
    return done


def resolve_revision(repo, revision):
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(repo, revision=revision).sha
    except Exception as e:  # offline etc.
        print(f"[warn] could not resolve commit hash: {e}")
        return revision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--prompts", default="prompts.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--outdir", default=".", help="folder for outputs and run_meta (e.g. a Google Drive folder)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts not yet done")
    ap.add_argument("--sample", action="store_true",
                    help="240-prompt sample: pair 0, both orders, repeat 0 of all 100 couple types + 40 baseline")
    ap.add_argument("--chunk", type=int, default=256, help="prompts per checkpoint")
    ap.add_argument("--revision", default=None, help="model commit hash; default = latest, recorded in run_meta")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--mm-off", action="store_true", help="disable image/audio inputs to save GPU memory")
    ap.add_argument("--qwen-thinking", action="store_true", help="keep Qwen's default thinking mode")
    args = ap.parse_args()

    cfg = dict(MODELS[args.model])
    if args.model == "qwen" and args.qwen_thinking:
        cfg.update(QWEN_THINKING)
    os.makedirs(args.outdir, exist_ok=True)
    out_path = args.out or os.path.join(args.outdir, f"outputs_{args.model}.jsonl")
    max_tokens = args.max_tokens or cfg["max_tokens"]

    prompts = [json.loads(l) for l in open(args.prompts)]
    done = load_done(out_path)
    todo = [p for p in prompts if p["prompt_id"] not in done]
    if args.sample:
        todo = [p for p in todo if (p["condition"] == "main" and p["pair_id"] == 0 and p["repeat"] == 0)
                or (p["condition"] == "baseline" and p["repeat"] < 20)]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(prompts)} prompts, {len(done)} already done, running {len(todo)}")
    if not todo:
        return

    import vllm
    from vllm import LLM, SamplingParams

    revision = resolve_revision(cfg["repo"], args.revision)
    llm_kwargs = dict(model=cfg["repo"], revision=revision, dtype=cfg["dtype"],
                      max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem, seed=0)
    if cfg["quantization"]:
        llm_kwargs["quantization"] = cfg["quantization"]
    if args.mm_off:
        llm_kwargs["limit_mm_per_prompt"] = cfg["mm_off"]
    llm = LLM(**llm_kwargs)

    if cfg["sampling"]:
        base = SamplingParams(**cfg["sampling"])
    else:
        base = llm.get_default_sampling_params()  # model's generation_config.json defaults
    print("sampling params:", base, "| chat_template_kwargs:", cfg["chat_template_kwargs"])

    meta_path = os.path.join(args.outdir, f"run_meta_{args.model}.json")
    meta = dict(model=args.model, repo=cfg["repo"], revision=revision, vllm_version=vllm.__version__,
                quantization=cfg["quantization"], dtype=cfg["dtype"], max_model_len=args.max_model_len,
                max_tokens=max_tokens, sampling_params=str(base),
                chat_template_kwargs=cfg["chat_template_kwargs"], qwen_thinking=args.qwen_thinking,
                host=platform.node(),
                started=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        import torch
        meta["torch_version"] = torch.__version__
        meta["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    runs = json.load(open(meta_path)) if os.path.exists(meta_path) else []
    runs.append(meta)
    json.dump(runs, open(meta_path, "w"), indent=2)

    t0, n_done, n_tok = time.time(), 0, 0
    with open(out_path, "a") as f:
        for i in range(0, len(todo), args.chunk):
            chunk = todo[i:i + args.chunk]
            convs = [[{"role": "user", "content": p["prompt"]}] for p in chunk]
            params = []
            for p in chunk:
                sp = base.clone()
                sp.seed = p["seed"]
                sp.max_tokens = max_tokens
                params.append(sp)
            outs = llm.chat(convs, sampling_params=params, use_tqdm=True,
                            chat_template_kwargs=cfg["chat_template_kwargs"])
            for p, o in zip(chunk, outs):
                c = o.outputs[0]
                f.write(json.dumps(dict(
                    prompt_id=p["prompt_id"], model=args.model, repo=cfg["repo"], revision=revision,
                    seed=p["seed"], text=c.text, finish_reason=c.finish_reason,
                    n_prompt_tokens=len(o.prompt_token_ids or []), n_output_tokens=len(c.token_ids),
                    time=time.strftime("%Y-%m-%d %H:%M:%S")), ensure_ascii=False) + "\n")
                n_tok += len(c.token_ids)
            f.flush()
            os.fsync(f.fileno())
            n_done += len(chunk)
            el = time.time() - t0
            eta = el / n_done * (len(todo) - n_done)
            print(f"[checkpoint] {n_done}/{len(todo)} saved | {n_tok / el:.0f} output tok/s | "
                  f"avg {n_tok / n_done:.0f} tok/response | ETA {eta / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
