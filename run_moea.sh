#!/bin/sh
# Lanza los 2 brazos (nsga3, rs) para las semillas dadas (por defecto 0-4), desligados de la sesión (setsid). Reanudable: salta lo terminado
# y retoma desde el último checkpoint. Uso: ./run_moea.sh [semillas...]   Diseño: docs/MOEA_DESIGN.md
# Ablación del objetivo por cluster (un solo objetivo de accuracy + tamaño): ABL=1 [ABL_EXTRA="evolution.pop_size=28"] lanza además el brazo nsga3g (v2: ABL_EXTRA="evolution.pop_size=28" para 28 direcciones con 2 objetivos)
# Variante B (evaluar en los nodos que entrenan): TAG=_b EXTRA="sampling.eval_holdout=False" ./run_moea.sh
SEEDS=${@:-0 1 2 3 4}
CONFIG=${CONFIG:-configs/cs_moea.yaml}   # v2: CONFIG=configs/cs_moea2.yaml TAG=2 ./run_moea.sh
TAG=${TAG:-}   # sufijo de la variante: salidas en runs/moea$TAG
ROOT=runs/moea$TAG
mkdir -p runs/moea_logs
for s in $SEEDS; do
  COMMON="seed=$s dataset.seed=$s probe.seed=$s evolution.seed=$s"
  launch() {
    name=$1; shift
    [ -f $ROOT/$name/seed$s/cost.json ] && return
    setsid nohup .venv/bin/python -u run_cs.py --config $CONFIG --output $ROOT/$name/seed$s --set $COMMON $EXTRA "$@" >> runs/moea_logs/${name}${TAG}_seed$s.log 2>&1 < /dev/null &
  }
  launch nsga3
  launch rs evolution.search=random
  [ -n "$ABL" ] && launch nsga3g evolution.objectives=cluster_mean_global_size $ABL_EXTRA
done
