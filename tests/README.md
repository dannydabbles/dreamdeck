# Test Suite Organization

- `smoke/`: Fast, isolated unit tests. Run frequently during development.
- `integration/`: Slower, multi-component, or LLM-involving tests. Run before merges or releases.

**Commands:**

- `make smoke` — run smoke tests only
- `make integration` — run integration tests only
- `make test` — run all tests
