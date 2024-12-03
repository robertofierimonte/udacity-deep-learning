#!make

help: ## Display this help screen
	@grep -h -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

pre-commit: ## Run the pre-commit over the entire repo
	@poetry run pre-commit run --all-files

setup: ## Install all the required Python dependencies and create a jupyter kernel for the project
	@poetry env use $(shell cat .python-version) && \
		poetry lock --no-update && \
		poetry install --sync && \
		poetry run python -m ipykernel install --user --name="udacity-deep-learning-venv"

bump-version: ## Update the model version. Must specify version=X.Y.Z
	@poetry run cz bump --files-only --check-consistency --no-verify --retry ${version}
