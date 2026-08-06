#!/usr/bin/env bash
# Run the project checks using the files supplied with the subject.
# The moulinette is deliberately not downloaded or executed.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOADS="${DOWNLOADS:-$HOME/Downloads}"
cd "$ROOT"

DATASETS_URL="https://cdn.intra.42.fr/document/document/54812/datasets_public.zip"
MOULINETTE_URL="https://cdn.intra.42.fr/document/document/54815/moulinette.zip"
CACHE_DIR="${CACHE_DIR:-data/.subject-cache}"
mkdir -p "$CACHE_DIR"

fetch() {
    local url="$1" destination="$2"
    if [[ ! -f "$destination" ]]; then
        echo "Downloading $url" >&2
        wget --quiet --show-progress --continue -O "$destination" "$url"
    fi
}

# The datasets and moulinette are downloaded from the official subject URLs.
fetch "$DATASETS_URL" "$CACHE_DIR/datasets_public.zip"
fetch "$MOULINETTE_URL" "$CACHE_DIR/moulinette.zip"
DATASETS_ZIP="$CACHE_DIR/datasets_public.zip"
VLLM_ZIP="${DOWNLOADS}/vllm-0.10.1.zip"
[[ -f "$VLLM_ZIP" ]] || VLLM_ZIP="data/vllm-0.10.1.zip"

# Always start with an empty corpus. This prevents unrelated repositories from
# accidentally remaining under data/raw and affecting the official score.
rm -rf data/raw
mkdir -p data/raw
if [[ ! -f "$VLLM_ZIP" ]]; then
    echo "vllm-0.10.1.zip not found in $DOWNLOADS or data/" >&2
    exit 1
fi
unzip -q -o "$VLLM_ZIP" -d data/raw

if [[ -f "$DATASETS_ZIP" ]]; then
    rm -rf data/datasets
    mkdir -p data/datasets
    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "$tmp_dir"' EXIT
    unzip -q -o "$DATASETS_ZIP" -d "$tmp_dir"
    cp -R "$tmp_dir/datasets_public/public/." data/datasets/
fi

if [[ ! -d data/raw ]] || [[ ! -d data/datasets ]]; then
    echo "Subject files not found. Put datasets_public.zip and vllm-0.10.1.zip in $DOWNLOADS." >&2
    exit 1
fi

run() {
    echo "+ $*" >&2
    "$@"
}

rm -rf data/processed data/output/search_results data/output/search_results_and_answer
run uv run python -m src index --max_chunk_size 2000

for scope in docs code; do
    unanswered="data/datasets/UnansweredQuestions/dataset_${scope}_public.json"
    answered="data/datasets/AnsweredQuestions/dataset_${scope}_public.json"
    output="data/output/search_results/UnansweredQuestions"

    [[ -f "$unanswered" ]] || { echo "Missing $unanswered" >&2; exit 1; }
    [[ -f "$answered" ]] || { echo "Missing $answered" >&2; exit 1; }

    run uv run python -m src search_dataset \
        --dataset_path "$unanswered" \
        --k 10 \
        --save_directory "$output"

    result="$output/$(basename "$unanswered")"
    run uv run python -m src evaluate \
        --student_search_results_path "$result" \
        --dataset_path "$answered"
done

# Run the official evaluator as well. It is downloaded above, never committed,
# and is intentionally kept separate from the project implementation.
MOULINETTE_DIR="$(mktemp -d)"
trap 'rm -rf "$MOULINETTE_DIR"' EXIT
unzip -q -o "$CACHE_DIR/moulinette.zip" -d "$MOULINETTE_DIR"
case "$(uname -s)" in
    Linux) moulinette="$MOULINETTE_DIR/moulinette-ubuntu" ;;
    *) moulinette="$MOULINETTE_DIR/moulinette-fedora" ;;
esac
chmod +x "$moulinette"
for scope in docs code; do
    result="data/output/search_results/UnansweredQuestions/dataset_${scope}_public.json"
    answered="data/datasets/AnsweredQuestions/dataset_${scope}_public.json"
    run "$moulinette" evaluate_student_search_results "$result" "$answered" \
        --k 10 --max_context_length 2000
done

# Set RUN_ANSWERS=1 to additionally download/use Qwen and test generation.
if [[ "${RUN_ANSWERS:-0}" == 1 ]]; then
    for scope in docs code; do
        run uv run python -m src answer_dataset \
            --student_search_results_path \
            "data/output/search_results/UnansweredQuestions/dataset_${scope}_public.json" \
            --save_directory "data/output/search_results_and_answer/UnansweredQuestions"
    done
fi

run uv run pytest -q
echo "All subject tests completed, including moulinette evaluation."
