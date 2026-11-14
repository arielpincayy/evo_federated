#!/bin/sh
# Lanza los 3 brazos de P3 para las semillas dadas (por defecto 0 1 2), desligados de la sesión (setsid). Reanudable: salta lo terminado
# y retoma desde el último checkpoint. Uso: ./run_p3.sh [semillas...]   Diseño: docs/P3_DESIGN.md
SEEDS=${@:-0 1 2}
mkdir -p runs/p3_logs
for s in $SEEDS; do
  COMMON="seed=$s dataset.seed=$s probe.seed=$s evolution.seed=$s"
  launch() {
    name=$1; shift
    [ -f runs/p3/$name/seed$s/cost.json ] && return
    setsid nohup .venv/bin/python -u run_cs.py --config configs/cs_p3.yaml --output runs/p3/$name/seed$s --set $COMMON "$@" >> runs/p3_logs/${name}_seed$s.log 2>&1 < /dev/null &
  }
  launch p3
  launch rs evolution.search=random
  launch base sampling.k_max=0 evolution.search=nsga2 evolution.objectives=mean_params evolution.pop_size=10 evolution.size_penalty=0 evolution.ref_partitions=2 final.hof=front_top final.max_front=12
done
