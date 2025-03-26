.PHONY: install run start build test lint format restart stop log aider-sonnet aider backup restore

# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

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

aider-dual:
	@echo "Running aider with local dual coder and reasoner models..."
	@aider --multiline --architect --model openai/reasoner --editor-model openai/coder --timeout 500 --model-settings-file .aider.model.settings.yml --test-cmd "make test" --auto-test --no-show-model-warnings

aider:
	@echo "Running aider with local llm..."
	@aider --multiline --architect --o1-mini --openai-api-base http://192.168.1.111:5000/v1 --timeout 500 --model-settings-file .aider.model.settings.yml

.PHONY: backup restore

backup:
	mkdir -p backups
	TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
	BACKUP_DIR := backups/dreamdeck_snapshot_$${TIMESTAMP}
	mkdir -p $$BACKUP_DIR
	# Backup PostgreSQL data
	cp -r ./.data/postgres $$BACKUP_DIR/postgres
	# Backup knowledge directory
	cp -r ./knowledge $$BACKUP_DIR/knowledge
	# Backup LocalStack S3 data
	cp -r ./my-localstack-data $$BACKUP_DIR/localstack
	# Backup ChromaDB data
	test -d ./chroma_db && cp -r ./chroma_db $$BACKUP_DIR/chroma_db || true
	# Collect app files and metadata
	cp config.yaml $$BACKUP_DIR/
	cp Dockerfile $$BACKUP_DIR/
	git rev-parse HEAD > $$BACKUP_DIR/git_commit_sha.txt
	echo "Backup Summary" > $$BACKUP_DIR/backup_summary.txt
	echo "Commit SHA: $$(git rev-parse HEAD)" >> $$BACKUP_DIR/backup_summary.txt
	echo "Timestamp: $${TIMESTAMP}" >> $$BACKUP_DIR/backup_summary.txt
	echo "Included directories:" >> $$BACKUP_DIR/backup_summary.txt
	echo "- postgres" >> $$BACKUP_DIR/backup_summary.txt
	echo "- knowledge" >> $$BACKUP_DIR/backup_summary.txt
	echo "- localstack" >> $$BACKUP_DIR/backup_summary.txt
	echo "- chroma_db (if exists)" >> $$BACKUP_DIR/backup_summary.txt
	echo "App files included: config.yaml, Dockerfile" >> $$BACKUP_DIR/backup_summary.txt
	tar -czvf $$BACKUP_DIR.tar.gz -C backups $$TIMESTAMP
	rm -rf $$BACKUP_DIR
	mv $$BACKUP_DIR.tar.gz backups/

restore:
	if [ -z "$$1" ]; then \
		LATEST_BACKUP=$$(ls -t backups/*.tar.gz | head -1); \
	else \
		LATEST_BACKUP=$$1; \
	fi; \
	mkdir -p restore_temp; \
	tar -xzvf $$$$LATEST_BACKUP -C restore_temp; \
	RESTORE_DIR=$$(find restore_temp -mindepth 1 -maxdepth 1 -type d); \

	# Move existing data to /tmp before restoring
	mv -f ./.data/postgres "/tmp/postgres_backup_$$(date +%s)" || true; \
	cp -r $$$$RESTORE_DIR/postgres ./.data/postgres; \

	mv -f ./knowledge "/tmp/knowledge_backup_$$(date +%s)" || true; \
	cp -r $$$$RESTORE_DIR/knowledge ./knowledge; \

	mv -f ./my-localstack-data "/tmp/localstack_backup_$$(date +%s)" || true; \
	cp -r $$$$RESTORE_DIR/localstack ./my-localstack-data; \

	mv -f ./chroma_db "/tmp/chroma_backup_$$(date +%s)" || true; \
	cp -r $$$$RESTORE_DIR/chroma_db ./chroma_db || true; \

	rm -rf restore_temp; \
	echo "Restored from $$$$LATEST_BACKUP"
