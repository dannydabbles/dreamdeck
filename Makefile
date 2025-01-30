.PHONY: install run start build test lint format restart stop log

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
