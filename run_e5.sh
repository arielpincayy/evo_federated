#!/bin/sh
# Lanza los 6 brazos de E5 para las semillas dadas (por defecto 0-4) desligados de la sesión (setsid).
# Es reanudable: si una corrida ya terminó se salta, y si se interrumpió retoma desde su último checkpoint
# (search_checkpoint.pkl por generación y final_progress.pkl en la etapa final). Uso: ./run_e5.sh [semillas...]
SEEDS=${@:-0 1 2 3 4}
mkdir -p runs/e5_logs
for s in $SEEDS; do
  COMMON="seed=$s dataset.seed=$s probe.seed=$s evolution.seed=$s"
  launch() {
    name=$1; shift
    [ -f runs/e5/$name/seed$s/cost.json ] && return
    setsid nohup .venv/bin/python -u run_cs.py --config configs/cs_e5.yaml --output runs/e5/$name/seed$s --set $COMMON "$@" >> runs/e5_logs/${name}_seed$s.log 2>&1 < /dev/null &
  }
  launch n3_cs
  launch n3_rand sampling.sampler=random
  launch n3_cs_lownoise evolution.objectives=per_cluster_mean
  launch rs_cs evolution.search=random
  launch n2_cs evolution.search=nsga2 evolution.objectives=mean_params evolution.pop_size=10 evolution.n_generations=20
  launch n2_rand sampling.sampler=random evolution.search=nsga2 evolution.objectives=mean_params evolution.pop_size=10 evolution.n_generations=20
done
