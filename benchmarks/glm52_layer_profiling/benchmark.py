# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from pathlib import Path

from benchmarks.glm52_layer_profiling.config import load_yaml, manifest, parse_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile GLM-5.2 layers 0..6")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--production-profile", action="store_true")
    parser.add_argument("--profile", choices=("none", "torch"))
    parser.add_argument("--profile-output-dir")
    args = parser.parse_args()
    data = load_yaml(args.config)
    if args.profile is not None:
        data["profile"] = args.profile
    if args.profile_output_dir is not None:
        data["profile_output_dir"] = args.profile_output_dir
    config = parse_config(data)
    if args.dry_run == args.production_profile:
        raise SystemExit("select exactly one of --dry-run and --production-profile")
    if args.dry_run:
        print(json.dumps(manifest(config), indent=2, sort_keys=True))
        return
    from benchmarks.glm52_layer_profiling.production_profile import run

    run(config)


if __name__ == "__main__":
    main()
