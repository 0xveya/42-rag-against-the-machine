SHELL := /bin/bash

UV ?= uv
PYTHON ?= python3
DOWNLOADS ?= $(HOME)/Downloads
CACHE_DIR ?= data/.subject-cache
MAX_CHUNK_SIZE ?= 2000
K ?= 10
QUERY ?=
DATASET_SCOPE ?= docs
DATASET_PATH ?= data/datasets/UnansweredQuestions/dataset_$(DATASET_SCOPE)_public.json
GROUND_TRUTH_PATH ?= data/datasets/AnsweredQuestions/dataset_$(DATASET_SCOPE)_public.json
SEARCH_OUTPUT_DIR ?= data/output/search_results/UnansweredQuestions
ANSWER_OUTPUT_DIR ?= data/output/search_results_and_answer/UnansweredQuestions
SEARCH_RESULTS_PATH ?= $(SEARCH_OUTPUT_DIR)/$(notdir $(DATASET_PATH))
MODEL ?= Qwen/Qwen3-0.6B
ARGS ?=

DATASETS_URL := https://cdn.intra.42.fr/document/document/54812/datasets_public.zip
MOULINETTE_URL := https://cdn.intra.42.fr/document/document/54815/moulinette.zip
DATASETS_ZIP := $(CACHE_DIR)/datasets_public.zip
MOULINETTE_ZIP := $(CACHE_DIR)/moulinette.zip
REFERENCE_URL := https://github.com/mpouillo/42-rag-against-the-machine
REFERENCE_DIR ?= /tmp/42-rag-reference
VLLM_ZIP := $(firstword $(wildcard $(DOWNLOADS)/vllm-0.10.1.zip data/vllm-0.10.1.zip))
MYPY_FLAGS := --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint lint-strict test index search \
	search-dataset answer answer-dataset evaluate subject-download subject-setup \
	subject-index subject-search moulinette moulinette-compare subject-eval \
	subject-all benchmark-repos benchmark-index

install:
	$(UV) sync --dev

run:
	$(UV) run python -m src $(ARGS)

debug:
	$(UV) run python -m pdb -m src $(ARGS)

index:
	$(UV) run python -m src index --max_chunk_size $(MAX_CHUNK_SIZE) $(ARGS)

search:
	@test -n "$(QUERY)" || { echo "Usage: make search QUERY='question' [K=10]" >&2; exit 2; }
	$(UV) run python -m src search "$(QUERY)" --k $(K) $(ARGS)

search-dataset:
	$(UV) run python -m src search_dataset \
		--dataset_path "$(DATASET_PATH)" --k $(K) \
		--save_directory "$(SEARCH_OUTPUT_DIR)" $(ARGS)

answer:
	@test -n "$(QUERY)" || { echo "Usage: make answer QUERY='question' [K=10]" >&2; exit 2; }
	$(UV) run python -m src answer "$(QUERY)" --k $(K) --model "$(MODEL)" $(ARGS)

answer-dataset:
	$(UV) run python -m src answer_dataset \
		--student_search_results_path "$(SEARCH_RESULTS_PATH)" \
		--save_directory "$(ANSWER_OUTPUT_DIR)" --model "$(MODEL)" $(ARGS)

evaluate:
	$(UV) run python -m src evaluate \
		--student_search_results_path "$(SEARCH_RESULTS_PATH)" \
		--dataset_path "$(GROUND_TRUTH_PATH)" $(ARGS)

subject-download:
	@mkdir -p "$(CACHE_DIR)"
	@test -f "$(DATASETS_ZIP)" || wget --quiet --show-progress --continue \
		-O "$(DATASETS_ZIP)" "$(DATASETS_URL)"
	@test -f "$(MOULINETTE_ZIP)" || wget --quiet --show-progress --continue \
		-O "$(MOULINETTE_ZIP)" "$(MOULINETTE_URL)"

subject-setup: subject-download
	@test -n "$(VLLM_ZIP)" || { \
		echo "Put vllm-0.10.1.zip in $(DOWNLOADS) or data/." >&2; exit 2; \
	}
	@rm -rf data/raw data/datasets
	@mkdir -p data/raw data/datasets
	@unzip -q -o "$(VLLM_ZIP)" -d data/raw
	@tmp=$$(mktemp -d); trap 'rm -rf "$$tmp"' EXIT; \
		unzip -q -o "$(DATASETS_ZIP)" -d "$$tmp"; \
		cp -R "$$tmp/datasets_public/public/." data/datasets/

subject-index: subject-setup
	@rm -rf data/processed
	@TIMEFORMAT='Indexing time: %3R seconds'; time \
		$(UV) run python -m src index --max_chunk_size $(MAX_CHUNK_SIZE)

subject-search: subject-index
	@rm -rf "$(SEARCH_OUTPUT_DIR)"
	@TIMEFORMAT='Docs search time: %3R seconds'; time \
		$(MAKE) --no-print-directory search-dataset DATASET_SCOPE=docs
	@TIMEFORMAT='Code search time: %3R seconds'; time \
		$(MAKE) --no-print-directory search-dataset DATASET_SCOPE=code

moulinette: subject-search
	@tmp=$$(mktemp -d); trap 'rm -rf "$$tmp"' EXIT; \
		unzip -q -o "$(MOULINETTE_ZIP)" -d "$$tmp"; \
		binary="$$tmp/moulinette-ubuntu"; \
		if grep -qi '^ID=fedora' /etc/os-release 2>/dev/null; then \
			binary="$$tmp/moulinette-fedora"; \
		fi; \
		chmod +x "$$binary"; \
		for scope in docs code; do \
			echo "=== Official $$scope evaluation ==="; \
			"$$binary" evaluate_student_search_results \
				"$(SEARCH_OUTPUT_DIR)/dataset_$${scope}_public.json" \
				"data/datasets/AnsweredQuestions/dataset_$${scope}_public.json" \
				--k $(K) --max_context_length $(MAX_CHUNK_SIZE); \
		done

moulinette-compare: subject-setup
	@set -euo pipefail; \
		tmp=$$(mktemp -d); trap 'rm -rf "$$tmp"' EXIT; \
		unzip -q -o "$(MOULINETTE_ZIP)" -d "$$tmp"; \
		moulinette="$$tmp/moulinette-ubuntu"; \
		if grep -qi '^ID=fedora' /etc/os-release 2>/dev/null; then \
			moulinette="$$tmp/moulinette-fedora"; \
		fi; chmod +x "$$moulinette"; \
		rm -rf data/processed "$(SEARCH_OUTPUT_DIR)"; \
		start=$$(date +%s%N); \
		$(UV) run python -m src index --max_chunk_size $(MAX_CHUNK_SIZE) \
			>/tmp/rag-mine-index.log 2>&1; \
		mine_index=$$((($$(date +%s%N)-start)/1000000)); \
		start=$$(date +%s%N); \
		for scope in docs code; do \
			$(UV) run python -m src search_dataset \
				--dataset_path data/datasets/UnansweredQuestions/dataset_$${scope}_public.json \
				--k $(K) --save_directory "$(SEARCH_OUTPUT_DIR)" \
				>/tmp/rag-mine-$${scope}.log 2>&1; \
		done; \
		mine_search=$$((($$(date +%s%N)-start)/1000000)); \
		mine_docs=$$("$$moulinette" evaluate_student_search_results \
			"$(SEARCH_OUTPUT_DIR)/dataset_docs_public.json" \
			data/datasets/AnsweredQuestions/dataset_docs_public.json \
			--k $(K) --max_context_length $(MAX_CHUNK_SIZE) | \
			awk '/Recall@5:/{print $$3}'); \
		mine_code=$$("$$moulinette" evaluate_student_search_results \
			"$(SEARCH_OUTPUT_DIR)/dataset_code_public.json" \
			data/datasets/AnsweredQuestions/dataset_code_public.json \
			--k $(K) --max_context_length $(MAX_CHUNK_SIZE) | \
			awk '/Recall@5:/{print $$3}'); \
		if [[ ! -d "$(REFERENCE_DIR)/.git" ]]; then \
			git clone --depth 1 "$(REFERENCE_URL)" "$(REFERENCE_DIR)"; \
		fi; \
		rm -rf "$(REFERENCE_DIR)/data"; \
		mkdir -p "$(REFERENCE_DIR)/data/output"; \
		ln -s "$(CURDIR)/data/raw" "$(REFERENCE_DIR)/data/raw"; \
		ln -s "$(CURDIR)/data/datasets" "$(REFERENCE_DIR)/data/datasets"; \
		cd "$(REFERENCE_DIR)"; env -u VIRTUAL_ENV $(UV) sync >/dev/null; \
		start=$$(date +%s%N); \
		env -u VIRTUAL_ENV $(UV) run python -m src index --max_chunk_size $(MAX_CHUNK_SIZE) \
			>/tmp/rag-reference-index.log 2>&1; \
		ref_index=$$((($$(date +%s%N)-start)/1000000)); \
		start=$$(date +%s%N); \
		for scope in docs code; do \
			env -u VIRTUAL_ENV $(UV) run python -m src search_dataset \
				--dataset_path data/datasets/UnansweredQuestions/dataset_$${scope}_public.json \
				--k $(K) --save_directory data/output \
				>/tmp/rag-reference-$${scope}.log 2>&1; \
		done; \
		ref_search=$$((($$(date +%s%N)-start)/1000000)); \
		ref_docs=$$("$$moulinette" evaluate_student_search_results \
			data/output/dataset_docs_public.json \
			data/datasets/AnsweredQuestions/dataset_docs_public.json \
			--k $(K) --max_context_length $(MAX_CHUNK_SIZE) | \
			awk '/Recall@5:/{print $$3}'); \
		ref_code=$$("$$moulinette" evaluate_student_search_results \
			data/output/dataset_code_public.json \
			data/datasets/AnsweredQuestions/dataset_code_public.json \
			--k $(K) --max_context_length $(MAX_CHUNK_SIZE) | \
			awk '/Recall@5:/{print $$3}'); \
		printf '\n| Implementation | Docs R@5 | Code R@5 | Index | Search (199) |\n'; \
		printf '|---|---:|---:|---:|---:|\n'; \
		printf '| This project | %s | %s | %.3fs | %.3fs |\n' \
			"$$mine_docs" "$$mine_code" "$$(awk "BEGIN{print $$mine_index/1000}")" \
			"$$(awk "BEGIN{print $$mine_search/1000}")"; \
		printf '| mpouillo reference | %s | %s | %.3fs | %.3fs |\n' \
			"$$ref_docs" "$$ref_code" "$$(awk "BEGIN{print $$ref_index/1000}")" \
			"$$(awk "BEGIN{print $$ref_search/1000}")"

subject-eval: moulinette
	$(UV) run pytest -q
	$(MAKE) --no-print-directory lint

subject-all: subject-eval

benchmark-repos:
	@if [[ "$(BENCHMARK_CLEAN)" == "1" ]]; then rm -rf data/raw; fi
	./scripts/clone_raw_repos.sh

benchmark-index: benchmark-repos
	@rm -rf data/processed
	@TIMEFORMAT='16-repository indexing time: %3R seconds'; time \
		$(UV) run python -m src index --max_chunk_size $(MAX_CHUNK_SIZE)

clean:
	find . -type d \( -name __pycache__ -o -name .mypy_cache \
		-o -name .ruff_cache -o -name .pytest_cache -o -name .ty \) \
		-prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	rm -rf build dist

lint:
	$(UV) run flake8 .
	$(UV) run mypy . $(MYPY_FLAGS)

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict

test:
	$(UV) run pytest -q
