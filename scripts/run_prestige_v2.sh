#!/usr/bin/env bash
# run_prestige_v2.sh — the redesigned prestige-weighted debate (prestige_mad_v2.py).
# prestige = running accuracy (warm-up W), confidence = stated. Policies: prestige, all_speak, none.
# Produces a store plus a D = confidence - prestige over-time plot (<store>_D.png).
#
#   bash scripts/run_prestige_v2.sh mock                    # no API calls, sanity check + D plot
#   bash scripts/run_prestige_v2.sh gemini                  # 3.5 Flash expert + 4x 3.1 Flash-Lite, GPQA-Diamond
#   bash scripts/run_prestige_v2.sh gpqa_full               # the pre-registered powered run: all 546 GPQA extended
#                                                        # questions, plus a report on the 176 untouched ones
#   bash scripts/run_prestige_v2.sh qwen                    # open models on a GPU (Qwen 7B + 4x 3B), MATH
#   bash scripts/run_prestige_v2.sh qwen_peers              # five different Qwen models of 4-8B (no designated
#                                                        # expert; prestige must find the best), MATH
#   bash scripts/run_prestige_v2.sh mixed                   # five open models of 4-8B from five companies, MATH
#   N=100 POLICIES=none bash scripts/run_prestige_v2.sh mixed   # pilot: round 1 only, to check the skills are close
#   bash scripts/run_prestige_v2.sh report <store.jsonl>    # accuracy + D plot only (free)
#
#   N=250 WARMUP=40 bash scripts/run_prestige_v2.sh gemini
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:-mock}
N=${N:-120}; WARMUP=${WARMUP:-20}; ROUNDS=${ROUNDS:-3}
POLICIES=${POLICIES:-prestige,all_speak,none}
MAD=prestige_mad_v2.py
MAD3=prestige_mad_v3.py        # open-model panel files (qwen_peers, mixed)

case "$STAGE" in
  mock)
    python3 $MAD --mock --n "${N:-60}" --warmup "$WARMUP" --rounds "$ROUNDS" \
      --policies "$POLICIES" --store runs/prestige_mock.jsonl ;;
  gemini)
    PROBLEMS=${PROBLEMS:-gpqa_diamond.jsonl}
    [[ -f "$PROBLEMS" ]] || python3 scripts/make_gpqa.py --subset diamond --out "$PROBLEMS"
    grep -qxF "gpqa*.jsonl" .gitignore 2>/dev/null || echo "gpqa*.jsonl" >> .gitignore
    python3 $MAD --design expert --expert-model gemini-3.5-flash --expert-thinking low \
      --weak-model gemini-3.1-flash-lite --weak-thinking none \
      --problems "$PROBLEMS" --n "$N" --warmup "$WARMUP" --rounds "$ROUNDS" \
      --workers 3 --stagger 0.45 --max-retries 8 --policies "$POLICIES" \
      --store "runs/prestige_gemini_$(basename "$PROBLEMS" .jsonl)_n${N}.jsonl" ;;
  qwen)
    PROBLEMS=${PROBLEMS:-math_extra.jsonl}
    [[ -f "$PROBLEMS" ]] || python3 scripts/make_math_extra.py --exclude math.jsonl --out "$PROBLEMS"
    bash scripts/serve_models.sh "Qwen/Qwen2.5-7B-Instruct 8001 0 0.45" "Qwen/Qwen2.5-3B-Instruct 8002 0 0.35"
    python3 $MAD --design expert_open --expert-model Qwen/Qwen2.5-7B-Instruct \
      --weak-model Qwen/Qwen2.5-3B-Instruct --problems "$PROBLEMS" --level 4,5 \
      --n "${N:-600}" --warmup "${WARMUP:-50}" --rounds "$ROUNDS" --workers 24 --stagger 0 \
      --max-retries 8 --max-tokens 2048 --policies "$POLICIES" \
      --store "runs/prestige_qwen_math45_n${N:-600}.jsonl"
    echo "stop the servers with: pkill -f 'vllm serve'" ;;
  qwen_peers)
    # Five DIFFERENT open Qwen models, each no bigger than 8B: two generations of 7B, a code-tuned 7B,
    # and Qwen3 4B and 8B (thinking off). Closer skills and less correlated errors than 7B + 4 copies of 3B,
    # and nobody is labelled the expert. Panel definition: panels/qwen_peers.json.
    # Memory: about 70 GB of weights in total. Default layout is two 80 GB GPUs; change GPU_A / GPU_B and the
    # MEM_* fractions for other cards (each 7-8B server needs roughly 18-24 GB including its cache).
    PROBLEMS=${PROBLEMS:-math_extra.jsonl}
    [[ -f "$PROBLEMS" ]] || python3 scripts/make_math_extra.py --exclude math.jsonl --out "$PROBLEMS"
    GPU_A=${GPU_A:-0}; GPU_B=${GPU_B:-1}; MEM7=${MEM7:-0.30}; MEM8=${MEM8:-0.35}; MEM4=${MEM4:-0.25}
    bash scripts/serve_models.sh \
      "Qwen/Qwen2-7B-Instruct 8011 $GPU_A $MEM7" \
      "Qwen/Qwen2.5-7B-Instruct 8012 $GPU_A $MEM7" \
      "Qwen/Qwen3-4B-Instruct-2507 8014 $GPU_A $MEM4" \
      "Qwen/Qwen2.5-Coder-7B-Instruct 8013 $GPU_B $MEM7" \
      "Qwen/Qwen3-8B 8015 $GPU_B $MEM8"
    python3 $MAD3 --panel-file panels/qwen_peers.json --problems "$PROBLEMS" --level 4,5 \
      --n "${N:-600}" --warmup "${WARMUP:-50}" --rounds "$ROUNDS" --workers 24 --stagger 0 \
      --max-retries 8 --max-tokens 2048 --policies "$POLICIES" \
      --store "runs/prestige_qwen_peers_math45_n${N:-600}.jsonl"
    echo "report: FRESH= bash scripts/run_prestige_v2.sh report runs/prestige_qwen_peers_math45_n${N:-600}.jsonl"
    echo "stop the servers with: pkill -f 'vllm serve'" ;;
  mixed)
    # Five open models from five different companies, each no bigger than 8B: Qwen3-8B (Alibaba, thinking off),
    # Ministral 3 8B (Mistral), Llama-3.1-Nemotron-Nano-8B (NVIDIA, thinking off), Gemma 3 4B (Google),
    # Phi-4-mini 3.8B (Microsoft). Different training data, so errors should overlap less than within one family.
    # Panel definition: panels/mixed_families.json. About 65 GB of weights; default layout is two 80 GB GPUs.
    PROBLEMS=${PROBLEMS:-math_extra.jsonl}
    [[ -f "$PROBLEMS" ]] || python3 scripts/make_math_extra.py --exclude math.jsonl --out "$PROBLEMS"
    GPU_A=${GPU_A:-0}; GPU_B=${GPU_B:-1}; MEM8=${MEM8:-0.32}; MEM4=${MEM4:-0.22}
    MISTRAL_FLAGS=${MISTRAL_FLAGS:---tokenizer-mode mistral --config-format mistral --load-format mistral}
    bash scripts/serve_models.sh \
      "Qwen/Qwen3-8B 8021 $GPU_A $MEM8" \
      "nvidia/Llama-3.1-Nemotron-Nano-8B-v1 8023 $GPU_A $MEM8" \
      "google/gemma-3-4b-it 8024 $GPU_A $MEM4" \
      "mistralai/Ministral-3-8B-Instruct-2512 8022 $GPU_B $MEM8 $MISTRAL_FLAGS" \
      "microsoft/Phi-4-mini-instruct 8025 $GPU_B $MEM4"
    python3 $MAD3 --panel-file panels/mixed_families.json --problems "$PROBLEMS" --level 4,5 \
      --n "${N:-600}" --warmup "${WARMUP:-50}" --rounds "$ROUNDS" --workers 24 --stagger 0 \
      --max-retries 8 --max-tokens 2048 --policies "$POLICIES" \
      --store "runs/prestige_mixed_math45_n${N:-600}$( [[ "$POLICIES" == none ]] && echo _pilot ).jsonl"
    echo "report: FRESH= bash scripts/run_prestige_v2.sh report runs/prestige_mixed_math45_n${N:-600}$( [[ "$POLICIES" == none ]] && echo _pilot ).jsonl"
    echo "stop the servers with: pkill -f 'vllm serve'" ;;
  gpqa_full)
    PROBLEMS=gpqa_extended.jsonl; N=546
    [[ -f "$PROBLEMS" ]] || python3 scripts/make_gpqa.py --subset extended --out "$PROBLEMS"
    grep -qxF "gpqa*.jsonl" .gitignore 2>/dev/null || echo "gpqa*.jsonl" >> .gitignore
    STORE="runs/prestige_gemini_gpqa_extended_n546.jsonl"
    ARGS=(--design expert --expert-model gemini-3.5-flash --expert-thinking low
          --weak-model gemini-3.1-flash-lite --weak-thinking none
          --problems "$PROBLEMS" --n "$N" --warmup "$WARMUP" --rounds "$ROUNDS"
          --workers "${WORKERS:-3}" --stagger "${STAGGER:-0.45}" --max-retries 8 --policies prestige,all_speak,none
          --store "$STORE" --fresh-against gpqa_main_extra.jsonl:250,gpqa_diamond.jsonl:120)
    echo "About 2,700 round-1 calls plus up to ~9,800 debate calls (two seatings per debate); roughly 1.5 h."
    echo "Resumable: if it stops, rerun this same command and it continues where it left off."
    read -r -p "Run it? [y/N] " ok; [[ "$ok" == "y" || "$ok" == "Y" ]] || exit 0
    [[ -f "$STORE" ]] && echo "continuing $STORE"
    if command -v caffeinate >/dev/null; then caffeinate -i python3 $MAD "${ARGS[@]}"
    else python3 $MAD "${ARGS[@]}"; fi ;;
  report)
    python3 $MAD3 report "${2:?give a store path}" --fresh-against "${FRESH-gpqa_main_extra.jsonl:250,gpqa_diamond.jsonl:120}" ;;
  *) echo "usage: bash scripts/run_prestige_v2.sh mock|gemini|qwen|qwen_peers|mixed|report <store>"; exit 1 ;;
esac
