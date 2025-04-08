# Contributing to Dreamdeck

Thank you for your interest in contributing to Dreamdeck! We welcome contributions of all kinds, including bug fixes, new features, documentation improvements, and more.

## How to Contribute

1. **Fork the Repository**

   Click the "Fork" button on GitHub and clone your fork locally:

   ```bash
   git clone https://github.com/your-username/dreamdeck.git
   cd dreamdeck
   ```

2. **Create a Branch**

   Create a new branch for your feature or fix:

   ```bash
   git checkout -b my-feature
   ```

3. **Set Up Your Environment**

   - Install [Poetry](https://python-poetry.org/)
   - Run `make install` to install dependencies
   - Use `make run` to start the app locally
   - Use `make test` to run tests

4. **Make Your Changes**

   - Follow existing code style and conventions
   - Add or update tests as needed
   - Update documentation if applicable

5. **Run Tests**

   Ensure all tests pass before submitting:

   ```bash
   make test
   ```

6. **Commit and Push**

   ```bash
   git add .
   git commit -m "Describe your changes"
   git push origin my-feature
   ```

7. **Open a Pull Request**

   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Describe your changes clearly
   - Link related issues if any

## Code Style

- Use [Black](https://black.readthedocs.io/) for formatting (`make format`)
- Follow PEP8 guidelines
- Write clear, concise docstrings
- Keep functions small and focused

## Reporting Issues

If you find a bug or have a feature request, please open an issue with:

- A clear title and description
- Steps to reproduce (if a bug)
- Expected vs actual behavior
- Screenshots or logs if helpful

## Community

Join our Discord or GitHub Discussions to chat with other contributors!

Thank you for helping make Dreamdeck better!
