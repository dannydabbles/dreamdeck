.PHONY: install run start build test lint format restart stop log aider-sonnet aider backup restore cli docker-test

# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate

install:
	@echo "Installing dependencies..."
	@echo "If using conda, ensure you have activated your environment (e.g., 'conda activate dd') before running this."
	@poetry install --with dev

run:
	@echo "Running the app locally..."
	@chainlit run src/app.py -w -h --port 8081 --debug

start:
	@echo "Starting the app with Docker..."
	@docker-compose up -d

build:
	@echo "Building the Docker image..."
	@docker-compose build

smoke:
	$(CONDA_ACTIVATE) dd && cd $(CURDIR) && PYTHONPATH=. poetry run pytest -v tests/smoke --cov=src --cov-report term-missing

integration:
	$(CONDA_ACTIVATE) dd && cd $(CURDIR) && PYTHONPATH=. poetry run pytest -v tests/integration --cov=src --cov-report term-missing

test:
	$(CONDA_ACTIVATE) dd && cd $(CURDIR) && PYTHONPATH=. poetry run pytest --tb=short tests/smoke tests/integration --cov=src --cov-report term-missing
	@echo "If you see dependency errors, try running 'conda activate dd' and then 'poetry install --with dev' to ensure dependencies are installed in your conda environment."

docker-test:
	@echo "Running tests in Docker/CI environment (no conda)..."
	PYTHONPATH=. poetry run pytest --tb=short tests/smoke tests/integration --cov=src --cov-report term-missing

lint:
	@echo "Linting the code..."
	@poetry run flake8 .

black:
	@echo "Formatting the code..."
	@poetry run black --diff .

format:
	@echo "Formatting the code with black..."
	@poetry run black . tests/

autofix:
	@echo "Automatically fixing imports with isort..."
	@poetry run isort .
	@echo "Automatically formatting code with black..."
	@poetry run black . tests/

restart: stop start
	@echo "Restarting the app with Docker..."

stop:
	@echo "Stopping the app..."
	@docker-compose down

log:
	@echo "Viewing logs in real-time..."
	@docker-compose logs -f

aider-gemini-gpt4.1:
	@echo "Running aider with gemini and gpt4.1..."
	@aider --multiline --architect --model openrouter/google/gemini-2.5-pro-exp-03-25 --editor-model gpt-4.1-2025-04-14 --weak-model gpt-4o-mini-2024-07-18 --timeout 500 --model-settings-file .aider.model.settings.yml --test-cmd "make test" --auto-test

aider-r1-local:
	@echo "Running aider with r1 and local models..."
	@aider --multiline --architect --model openrouter/deepseek/deepseek-r1 --editor-model openai/coder --timeout 500 --model-settings-file .aider.model.settings.yml --test-cmd "make test" --auto-test

aider-r1-gpt4.1:
	@echo "Running aider with r1 and gpt4.1..."
	@aider --multiline --architect --model openrouter/deepseek/deepseek-r1 --editor-model gpt-4.1-2025-04-14 --timeout 500 --model-settings-file .aider.model.settings.yml --test-cmd "make test" --auto-test

aider-r1-gpt4.1-mini:
	@echo "Running aider with r1 and gpt4.1 Mini..."
	@aider --multiline --architect --model openrouter/deepseek/deepseek-r1 --editor-model gpt-4o-mini-2024-07-18 --timeout 500 --model-settings-file .aider.model.settings.yml --test-cmd "make test" --auto-test

aider-gpt4.1:
	@echo "Running aider with gpt4.1..."
	@aider --multiline --architect --model gpt-4.1-2025-04-14 --timeout 500 --model-settings-file .aider.model.settings.yml --test-cmd "make test" --auto-test

aider-gemma-gpt4.1:
	@echo "Running aider with gemma and gpt4.1..."
	@aider --multiline --architect --model openrouter/google/gemini-2.5-pro-exp-03-25:free --editor-model gpt-4.1-2025-04-14 --test-cmd "make test" --auto-test

aider-gemma-local:
	@echo "Running aider with gemma and local models..."
	@aider --multiline --architect --model openrouter/google/gemini-2.5-pro-exp-03-25:free --editor-model openai/coder --timeout 900 --model-settings-file .aider.model.settings.yml --test-cmd "make test" --auto-test --no-show-model-warnings

aider-sonnet:
	@echo "Running aider with sonnet..."
	@aider --multiline --architect --sonnet --timeout 500 --model-settings-file .aider.model.settings.yml --test-cmd "make test" --auto-test --no-show-model-warnings

aider-optimus:
	@echo "Running aider with optimus..."
	@aider --multiline --architect --model openrouter/openrouter/optimus-alpha --editor-model openrouter/openrouter/optimus-alpha --timeout 500 --model-settings-file .aider.model.settings.yml --test-cmd "make test" --auto-test --no-show-model-warnings --editor-edit-format editor-diff --edit-format diff

aider-gemma-sonnet:
	@echo "Running aider with gemma and sonnet..."
	@aider --multiline --architect --model openrouter/google/gemini-2.5-pro-exp-03-25:free --editor-model sonnet --timeout 500 --model-settings-file .aider.model.settings.yml --test-cmd "make test" --auto-test --no-show-model-warnings

aider-gemma-optimus:
	@echo "Running aider with gemma and optimus..."
	@aider --multiline --architect --model openrouter/google/gemini-2.5-pro-exp-03-25:free --editor-model openrouter/openrouter/optimus-alpha --timeout 500 --model-settings-file .aider.model.settings.yml --test-cmd "make test" --auto-test --no-show-model-warnings --editor-edit-format editor-diff --edit-format diff

aider-gemma-qwen:
	@echo "Running aider with gemma and qwen..."
	@aider --multiline --architect --model openrouter/google/gemini-2.5-pro-exp-03-25:free --editor-model openai/coder --timeout 500 --model-settings-file .aider.model.settings.yml --test-cmd "make test" --auto-test --no-show-model-warnings

aider-dual:
	@echo "Running aider with local dual coder and reasoner models..."
	@aider --multiline --architect --model openai/reasoner --editor-model openai/coder --timeout 500 --model-settings-file .aider.model.settings.yml --test-cmd "make test" --auto-test --no-show-model-warnings

aider-local:
	@echo "Running aider with local models..."
	@aider --multiline --architect --model openai/coder --editor-model openai/coder --timeout 500 --model-settings-file .aider.model.settings.yml --test-cmd "make test" --auto-test --no-show-model-warnings

aider:
	@echo "Running aider with local llm..."
	@aider --multiline --architect --o1-mini --openai-api-base http://192.168.1.111:5000/v1 --timeout 500 --model-settings-file .aider.model.settings.yml

cli:
	@echo "Running Dreamdeck CLI..."
	@python3 -m src.cli

backup:
	mkdir -p backups; \
	TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	SNAPSHOT_DIR=dreamdeck_snapshot_"$$TIMESTAMP"; \
	BACKUP_DIR=backups/$$SNAPSHOT_DIR; \
	mkdir -p "$$BACKUP_DIR"; \
	cp -r ./.data/postgres "$$BACKUP_DIR/postgres"; \
	cp -r ./knowledge "$$BACKUP_DIR/knowledge"; \
	cp -r ./my-localstack-data "$$BACKUP_DIR/localstack"; \
	test -d ./chroma_db && cp -r ./chroma_db "$$BACKUP_DIR/chroma_db" || true; \
	cp config.yaml "$$BACKUP_DIR/"; \
	cp Dockerfile "$$BACKUP_DIR/"; \
	git rev-parse HEAD > "$$BACKUP_DIR/git_commit_sha.txt"; \
	echo "Backup Summary" > "$$BACKUP_DIR/backup_summary.txt"; \
	echo "Commit SHA: $$(git rev-parse HEAD)" >> "$$BACKUP_DIR/backup_summary.txt"; \
	echo "Timestamp: $$TIMESTAMP" >> "$$BACKUP_DIR/backup_summary.txt"; \
	echo "Included directories:" >> "$$BACKUP_DIR/backup_summary.txt"; \
	echo "- postgres" >> "$$BACKUP_DIR/backup_summary.txt"; \
	echo "- knowledge" >> "$$BACKUP_DIR/backup_summary.txt"; \
	echo "- localstack" >> "$$BACKUP_DIR/backup_summary.txt"; \
	echo "- chroma_db (if exists)" >> "$$BACKUP_DIR/backup_summary.txt"; \
	echo "App files included: config.yaml, Dockerfile" >> "$$BACKUP_DIR/backup_summary.txt"; \
	tar -czvf "$$SNAPSHOT_DIR.tar.gz" -C backups "$$SNAPSHOT_DIR"; \
	mv "$$SNAPSHOT_DIR.tar.gz" backups/; \
	mv "$$BACKUP_DIR" /tmp/;

restore:
	mkdir -p restore_temp; \
	if [ -n "$$RESTORE_FILE" ]; then \
		LATEST_BACKUP="$$RESTORE_FILE"; \
	else \
		LATEST_BACKUP=$$(ls -t backups/*.tar.gz | head -1); \
	fi; \
	tar -xzvf "$$LATEST_BACKUP" -C restore_temp; \
	RESTORE_DIR=$$(find restore_temp -mindepth 1 -maxdepth 1 -type d); \
	mv -f ./.data/postgres "/tmp/postgres_backup_$$(date +%s)" || true; \
	mv -f ./knowledge "/tmp/knowledge_backup_$$(date +%s)" || true; \
	mv -f ./my-localstack-data "/tmp/localstack_backup_$$(date +%s)" || true; \
	mv -f ./chroma_db "/tmp/chroma_backup_$$(date +%s)" || true; \
	cp -r "$$RESTORE_DIR/postgres" ./.data/postgres; \
	cp -r "$$RESTORE_DIR/knowledge" ./knowledge; \
	cp -r "$$RESTORE_DIR/localstack" ./my-localstack-data; \
	cp -r "$$RESTORE_DIR/chroma_db" ./chroma_db || true; \
	mv restore_temp /tmp/; \
	echo "Restored from $$LATEST_BACKUP";
# Add this new target for CI
ci-test:
	PYTHONPATH=. poetry run pytest --tb=short tests/smoke tests/integration --cov=src --cov-report term-missing
