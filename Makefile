.PHONY: install run start build test lint format restart stop log aider-sonnet aider backup restore

# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate

install:
	@echo "Installing dependencies..."
	@poetry install

run:
	@echo "Running the app locally..."
	@chainlit run src/app.py -w -h --port 8081 --debug

start:
	@echo "Starting the app with Docker..."
	@docker-compose up -d

build:
	@echo "Building the Docker image..."
	@docker-compose build

test:
	$(CONDA_ACTIVATE) dd && cd $(CURDIR) && PYTHONPATH=. poetry run pytest -v tests/ --cov=src --cov-report term-missing

lint:
	@echo "Linting the code..."
	@poetry run flake8 .

black:
	@echo "Formatting the code..."
	@poetry run black --diff .

format: lint
	@echo "Formatting the code..."
	@poetry run black src/

restart: stop start
	@echo "Restarting the app with Docker..."

stop:
	@echo "Stopping the app..."
	@docker-compose down

log:
	@echo "Viewing logs in real-time..."
	@docker-compose logs -f

aider-sonnet:
	@echo "Running aider with sonnet..."
	@aider --multiline --architect --sonnet

aider-gemma-sonnet:
	@echo "Running aider with gemma and sonnet..."
	@aider --multiline --architect --model openrouter/google/gemini-2.5-pro-exp-03-25:free --editor-model sonnet --timeout 500 --model-settings-file .aider.model.settings.yml --test-cmd "make test" --auto-test --no-show-model-warnings

aider-gemma-quasar:
	@echo "Running aider with gemma and quasar..."
	@aider --multiline --architect --model openrouter/google/gemini-2.5-pro-exp-03-25:free --editor-model openrouter/openrouter/quasar-alpha --timeout 500 --model-settings-file .aider.model.settings.yml --test-cmd "make test" --auto-test --no-show-model-warnings

aider-gemma-qwen:
	@echo "Running aider with gemma and qwen..."
	@aider --multiline --architect --model openrouter/google/gemini-2.5-pro-exp-03-25:free --editor-model openai/coder --timeout 500 --model-settings-file .aider.model.settings.yml --test-cmd "make test" --auto-test --no-show-model-warnings

aider-dual:
	@echo "Running aider with local dual coder and reasoner models..."
	@aider --multiline --architect --model openai/reasoner --editor-model openai/coder --timeout 500 --model-settings-file .aider.model.settings.yml --test-cmd "make test" --auto-test --no-show-model-warnings

aider:
	@echo "Running aider with local llm..."
	@aider --multiline --architect --o1-mini --openai-api-base http://192.168.1.111:5000/v1 --timeout 500 --model-settings-file .aider.model.settings.yml

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
