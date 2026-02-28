# A helper function to generate OOD datasets for evaluation

import os
import sys
import argparse

_this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(_this_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "fairseq"))

from data.pure_tasks import get_pure_dataset


OOD_SAMPLES = {
    256: 500,
    512: 250,
}
DEFAULT_SAMPLES = 500

TASK_DATASET_DIRS = {
    "positional": "datasets/positional",
    "semantic":   "datasets/semantic",
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate OOD datasets at requested sizes and cache to disk"
    )
    parser.add_argument("--task", type=str, required=True,
                        help="Task type (tnm, positional, semantic, ...)")
    parser.add_argument("--n_values", type=int, nargs="+", required=True,
                        help="Graph sizes to generate (e.g. 256 512)")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Output directory (default: datasets/{task}/)")
    parser.add_argument("--seed", type=int, default=77777,
                        help="RNG seed for dataset generation (default: 77777)")
    parser.add_argument("--samples", type=int, default=0,
                        help="Samples per dataset (0 = use per-size defaults: 500 for n<=256, 250 for n=512)")
    args = parser.parse_args()

    if not args.output_dir:
        task_dir = TASK_DATASET_DIRS.get(args.task, f"datasets/{args.task}")
        args.output_dir = os.path.join(root_dir, task_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Task:       {args.task}")
    print(f"Sizes:      {args.n_values}")
    print(f"Output dir: {args.output_dir}")
    print(f"Seed:       {args.seed}")

    for n in args.n_values:
        samples = args.samples if args.samples > 0 else OOD_SAMPLES.get(n, DEFAULT_SAMPLES)
        out_path = os.path.join(args.output_dir, f"n{n}_ood.pt")

        if os.path.exists(out_path):
            print(f"\n[SKIP] n={n}: cache already exists at {out_path}")
            continue

        print(f"\n[GEN] n={n} | {samples} samples | seed={args.seed} -> {out_path}")
        dataset = get_pure_dataset(
            args.task,
            num_nodes=n,
            num_samples=samples,
            seed=args.seed,
            cache_path=out_path,
        )
        print(f"  Done: {len(dataset)} samples saved to {out_path}")


if __name__ == "__main__":
    main()
