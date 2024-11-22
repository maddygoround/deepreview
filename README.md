# DeepReview - AI-Powered Code Review Tool

DeepReview is an intelligent code review assistant that uses local LLMs (Large Language Models) to analyze git branch differences and provide comprehensive code reviews. It helps developers identify potential issues, maintain code quality, and ensure best practices across their codebase.

## 🚀 Features

- Automated code review using local LLMs through Ollama
- Git branch difference analysis
- Detailed feedback on code quality, security, and performance
- Customizable review focus areas
- Support for multiple programming languages
- Markdown and YAML report generation

## 🛠️ Installation

1. Install the package:
```bash
pip install .
```

2. Install Ollama from [ollama.ai](https://ollama.ai)

3. Pull the required model:
```bash
ollama pull codellama
```

## 📖 Usage

Run from your git repository:
```bash
deepreview
```

## ⚙️ Configuration Guide

Create a `config.yaml` file in:
- Your project directory: `./config.yaml`

### Configuration Parameters Explained


## Configuration Reference

### LLM Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| `llm_model` | The Ollama model to use | `codellama` |
| `main_branch` | Your main branch name | `main` |

### File Exclusions

| Pattern | Description | Default |
|---------|-------------|---------|
| `ignore_files` | List of files to ignore | `[".lock", "package-lock.json", "yarn.lock", ".gitignore", "*.md", "*.log", "node_modules/**", "venv/**"]` |

### Analysis Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_diff_size` | Maximum diff size to analyze (bytes) | `100000` |
| `min_changes` | Minimum changes needed for analysis | `1` |

### Review Priority Weights

| Weight | Value | Description |
|--------|-------|-------------|
| `security` | `0.3` | Security considerations |
| `performance` | `0.25` | Performance implications |
| `code_quality` | `0.25` | Code quality and style |
| `architecture` | `0.2` | Architectural considerations |

### Language-Specific Settings

#### Python
| Setting | Description | Default |
|---------|-------------|---------|
| `check_typing` | Check type hints | `true` |
| `pep8_compliance` | Check PEP 8 style guide | `true` |

#### JavaScript
| Setting | Description | Default |
|---------|-------------|---------|
| `check_eslint` | Check ESLint rules | `true` |
| `modern_features` | Check modern JS features | `true` |

#### TypeScript
| Setting | Description | Default |
|---------|-------------|---------|
| `strict_mode` | Check TypeScript strict mode | `true` |
| `interface_check` | Check interface implementations | `true` |

### Output Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `format` | Output format (`markdown` or `yaml`) | `markdown` |
| `include_stats` | Include change statistics | `true` |
| `include_suggestions` | Include improvement suggestions | `true` |


## 📊 Output

DeepReview generates analysis reports covering:
1. Impact Analysis
   - Component/functionality effects
   - Breaking change detection

2. Code Quality Review
   - Style and best practices
   - Potential bugs
   - Design patterns

3. Security Assessment
   - Vulnerability detection
   - Auth/authorization review
   - Data handling analysis

4. Performance Review
   - Performance implications
   - Resource usage
   - Scalability concerns

5. Testing Recommendations
   - Required test types
   - Edge cases
   - Integration scenarios

Reports are saved in `./analysis_results` by default.

## 📋 Requirements

- Python 3.7+
- Git
- Ollama
- Local LLM (default: codellama)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License.
This README provides:
A clear introduction to the project
Detailed installation steps
Usage instructions
A comprehensive configuration guide with explanations for all parameters
Description of the analysis output
Project requirements
Information about contributing
The configuration section is particularly detailed, explaining each parameter in the config.yaml file to help users customize the tool for their needs.