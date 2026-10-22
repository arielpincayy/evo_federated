#!/usr/bin/env python
"""Ejecuta una corrida CS-NAS. Ej: python run_cs.py --config configs/cs_smoke.yaml --set sampling.sampler=random"""
import argparse
import ast
import dataclasses

from evofederated.experiments.cs_runner import run_cs
from evofederated.utils.config import _merge, from_dict, load_config


def _set(d: dict, dotted: str, raw: str):
    *path, last = dotted.split(".")
    for p in path:
        d = d.setdefault(p, {})
    try:
        d[last] = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        d[last] = raw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output")
    ap.add_argument("--set", nargs="*", default=[], help="override dotted, p. ej. evolution.seed=7")
    a = ap.parse_args()
    over = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        _set(over, k, v)
    cfg = from_dict(_merge(dataclasses.asdict(load_config(a.config)), over))
    print(run_cs(cfg, a.output))
