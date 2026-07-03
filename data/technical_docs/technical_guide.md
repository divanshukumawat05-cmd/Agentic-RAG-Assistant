# Internal Technical Guide

## Python Setup

All development work for this project should be completed using Python 3.10 or later. A virtual environment is required for every local setup to prevent dependency conflicts.

Recommended steps:

1. Create a virtual environment in the project root.
2. Activate the virtual environment.
3. Install dependencies from the requirements file.
4. Verify the installed version of Python before starting development.

Use the latest stable package versions whenever possible. Avoid installing packages globally unless there is a documented need to do so.

## VS Code Setup

Visual Studio Code is the recommended editor for this project. Install the following extensions for a better development experience:

- Python
- Pylance
- GitLens
- Docker
- Markdown All in One

Set the workspace to use the project virtual environment. Configure the editor to auto-format Python files and enable linting for improved code quality. Keep the integrated terminal open for running tests and deployment commands.

## Git Installation

Git must be installed on all developer machines before contributing to the repository. Confirm the installation by checking the version from the terminal.

Before beginning work, configure user identity with the developer's full name and company email. Always pull the latest changes from the main branch before creating a new feature branch.

## Docker Installation

Docker is used for containerized development and deployment. Install Docker Desktop or the appropriate Docker Engine package for the operating system.

Before running containers, verify that Docker is available and that the daemon is running. Use container images that match the project's runtime requirements and avoid personal or unapproved images.

## Project Folder Structure

The repository follows a modular layout for clarity and maintainability:

- data/ - input documents, sample databases, and reference datasets
- notebooks/ - exploratory analysis and proof-of-concept notebooks
- src/ - application source code
- src/agents/ - domain-specific agent logic
- src/database/ - database models and database access helpers
- src/rag/ - retrieval and embedding logic
- src/router/ - routing and orchestration logic
- src/utils/ - shared utility functions
- tests/ - automated test cases
- vector_db/ - vector storage artifacts

Maintain this structure unless a documented architecture change is approved.

## Running the Project

To run the application locally, ensure that the virtual environment is active and the required packages are installed. Use the application entry point from the source directory and run it in development mode.

The development workflow should include:

1. Starting the application with the local development command
2. Verifying that the API or web interface loads successfully
3. Testing the basic flows for document retrieval and agent interaction
4. Stopping the service cleanly after validation

Any environment variables or configuration values must be stored securely and must not be committed to the repository.

## Coding Standards

All code should follow the company's development standards:

- Use descriptive names for variables, classes, and functions
- Keep functions focused and avoid large monolithic blocks
- Write readable and maintainable code
- Add comments only where they improve clarity
- Follow consistent indentation and formatting
- Avoid hard-coded secrets or credentials

Prefer simple, well-tested implementations over unnecessarily complex solutions. Reuse existing utility modules when appropriate.

## Branch Naming

Use the following branch naming convention:

- feature/short-description
- bugfix/issue-number-short-description
- hotfix/urgent-fix-description
- chore/maintenance-task

Branch names must be lowercase and use hyphens instead of spaces.

## Commit Message Rules

Commit messages must be concise, descriptive, and written in the imperative mood. Use a short summary line followed by optional details if needed.

Example format:

- Add document retrieval fallback for vector search
- Fix mentor assignment mapping in HR agent
- Refactor router error handling for failed requests

Do not commit generated files, sensitive credentials, or large binary artifacts unless explicitly approved.

## API Development Guidelines

When creating or updating APIs:

- Keep endpoints consistent and predictable
- Use clear request and response schemas
- Return meaningful error messages
- Validate all incoming data
- Handle missing or invalid inputs gracefully
- Document major changes in the relevant repository notes

Use versioning where required and ensure backward compatibility whenever possible.

## Testing Guidelines

All new functionality should be covered by tests before being merged. At a minimum, validate the primary success path and the main failure path.

Testing should include:

- Unit tests for business logic
- Integration tests for database and retrieval behavior
- Validation for edge cases and input errors

Tests must be reproducible and should not depend on manual environment changes. Any failing test should be investigated before the change is considered complete.

## Deployment Notes

Deployment should be done only through approved environments such as development, testing, or production. Before deployment, confirm that all tests pass and that the build is clean.

Review the environment configuration, secret management process, and logging setup before release. Use rollback procedures if the deployment introduces regressions or service instability.
