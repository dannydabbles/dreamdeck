.PHONY: install run start build test lint format restart stop log aider-sonnet aider

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
	$(CONDA_ACTIVATE) dd && PYTHONPATH=. poetry run pytest

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
	@aider --multiline --architect --4o --openai-api-base http://192.168.1.111:5000/v1 --timeout 500 --model-settings-file .aider.model.settings.yml --model openai/reasoner --editor-model openai/coder --test-cmd "make test" --auto-test --no-show-model-warnings

aider:
	@echo "Running aider with local llm..."
	@aider --multiline --architect --o1-mini --openai-api-base http://192.168.1.111:5000/v1 --timeout 500 --model-settings-file .aider.model.settings.yml
