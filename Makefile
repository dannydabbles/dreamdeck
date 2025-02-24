.PHONY: install run start build test lint format restart stop log aider-sonnet aider

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
	@echo "Running tests..."
	@poetry run pytest

lint:
	@echo "Linting the code..."
	@poetry run flake8 src/

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

aider:
	@echo "Running aider with local llm..."
	@aider --multiline --architect --4o --openai-api-base http://192.168.1.111:5000/v1 --timeout 500 --model-settings-file .aider.model.settings.yml --model openai/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview-exl2 --editor-model openai/Qwen2.5-Coder-32B-Instruct-8.0bpw-exl2 --editor-edit-format editor-whole


