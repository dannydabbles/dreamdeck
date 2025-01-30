.PHONY: install run

install:
	@echo "Installing dependencies..."
	@poetry install

run:
	@echo "Running the app..."
	@chainlit run src/app.py -w -h --port 8081 --debug

all: run
