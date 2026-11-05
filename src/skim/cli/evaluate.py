from __future__ import annotations

import argparse
from pathlib import Path

import torch

from skim.config import EVAL_DECODING_MODES, EVAL_MAX_NEW_TOKENS, MODEL_REGISTRY
from skim.eval.metrics import headline
from skim.eval.run import DATASETS, evaluate_dataset
from skim.io import load_model, load_tokenizer, save_json
from skim.prune.masks import MaskApplier


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_REGISTRY))
    source = p.add_mutually_exclusive_group()
    source.add_argument("--model_path")
    source.add_argument("--mask_path")
    p.add_argument("--dataset", choices=list(DATASETS), required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval_max_new_tokens", type=int, default=EVAL_MAX_NEW_TOKENS)
    p.add_argument("--eval_decoding_modes", default=",".join(EVAL_DECODING_MODES))
    p.add_argument("--eval_limit", type=int, default=0)
    args = p.parse_args(argv)
    if not args.model and not args.model_path:
        p.error("--model is required unless --model_path is given")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    modes = tuple(m.strip() for m in args.eval_decoding_modes.split(",") if m.strip())

    source = {"model": args.model, "model_name": MODEL_REGISTRY.get(args.model)}
    if args.model_path:
        model_dir = Path(args.model_path).expanduser().resolve()
        if not (model_dir / "config.json").exists():
            raise ValueError(f"model path does not contain config.json: {model_dir}")
        model_dir = str(model_dir)
        method, source["model_path"] = "distilled", model_dir
    else:
        model_dir = source["model_name"]
        method = "pruned" if args.mask_path else "dense"
    tokenizer = load_tokenizer(model_dir)
    model = load_model(model_dir, args.device)
    if args.mask_path:
        masks = torch.load(args.mask_path, map_location="cpu")
        MaskApplier.apply(model, masks)
        MaskApplier.assert_applied(model, masks)
        source["mask_path"] = str(Path(args.mask_path).resolve())

    metrics = evaluate_dataset(
        model,
        tokenizer,
        args.dataset,
        dump_path=out / f"eval_dump_{args.dataset}.jsonl",
        modes=modes,
        max_new_tokens=args.eval_max_new_tokens,
        eval_limit=args.eval_limit,
    )
    payload = {
        "method": method,
        **source,
        "dataset": args.dataset,
        **headline(metrics),
        "metrics": metrics,
        "eval_config": {
            "decoding_modes": list(modes),
            "max_new_tokens": args.eval_max_new_tokens,
            "eval_limit": args.eval_limit,
            "device": args.device,
            "dtype": str(next(model.parameters()).dtype),
        },
    }
    save_json(out / f"post_{args.dataset}.json", payload)
    errors = {mode: metrics[mode]["overall"]["n_error"] for mode in metrics}
    print(
        f"[{method}] {args.dataset} {headline(metrics)} n_error={errors} "
        f"-> {out}/post_{args.dataset}.json",
        flush=True,
    )


if __name__ == "__main__":
    main()
