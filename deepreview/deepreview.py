import os
import subprocess
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
from datetime import datetime
import yaml
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import git

class BranchDiffAnalyzer:
    def __init__(self, config_path: str = "config.yaml"):
         # Look for config in current directory first, then home directory
        local_config = os.path.join(os.getcwd(), config_path)
        home_config = os.path.join(os.path.expanduser("~"), ".deepreview", config_path)
        
        if os.path.exists(local_config):
            self.config = self._load_config(local_config)
        elif os.path.exists(home_config):
            self.config = self._load_config(home_config)
        else:
            self.config = self._load_config(None)  # Load defaults
            
        try:
            self.repo = git.Repo(os.getcwd(), search_parent_directories=True)
        except git.exc.InvalidGitRepositoryError:
            raise Exception("Not a git repository. Please run this command from within a git repository.")
        
        self.llm = Ollama(model=self.config.get("llm_model", "codellama"))
        self.setup_logging()

    def setup_logging(self):
            # Create logs directory in user's home
        log_dir = os.path.join(os.path.expanduser("~"), ".deepreview", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'branch_analysis.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path: str) -> dict:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {
            "llm_model": "codellama",
            "main_branch": "main",
            "ignore_files": [".lock", ".json", ".md"],
            "max_diff_size": 100000,
        }

    def get_branch_diff(self, branch_name: Optional[str] = None, 
                       main_branch: Optional[str] = None) -> Dict[str, str]:
        """Get differences between current/specified branch and main"""
        if not branch_name:
            branch_name = self.repo.active_branch.name
        if not main_branch:
            main_branch = self.config.get("main_branch", "main")

        self.logger.info(f"Analyzing diff between {branch_name} and {main_branch}")
        
        try:
            # Get the diff between branches
            tree_a = self.repo.commit(main_branch).tree
            tree_b = self.repo.commit(branch_name).tree

            # Perform the diff between the trees
            diff_index = tree_a.diff(tree_b)
            
            changes = {}
            for diff_item in diff_index:
                # Skip ignored files
                if any(ignore in diff_item.a_path for ignore in self.config.get("ignore_files", [])):
                    continue
                
                # Get the full diff content
                diff_content = self.repo.git.diff(
                    f"{main_branch}...{branch_name}",
                    diff_item.a_path
                )
                
                changes[diff_item.a_path] = {
                    'content': diff_content,
                    'status': self._get_change_type(diff_item),
                    'stats': self._get_change_stats(diff_content)
                }
            
            return changes
        except git.exc.GitCommandError as e:
            self.logger.error(f"Git error: {e}")
            return {}

    def _get_change_type(self, diff_item) -> str:
        """Determine the type of change"""
        if diff_item.new_file:
            return 'added'
        elif diff_item.deleted_file:
            return 'deleted'
        elif diff_item.renamed:
            return 'renamed'
        else:
            return 'modified'

    def _get_change_stats(self, diff_content: str) -> Dict:
        """Get statistics about the changes"""
        lines = diff_content.split('\n')
        additions = len([l for l in lines if l.startswith('+')])
        deletions = len([l for l in lines if l.startswith('-')])
        return {
            'additions': additions,
            'deletions': deletions,
            'total_changes': additions + deletions
        }

    def analyze_changes(self, branch_name: Optional[str] = None) -> Dict:
        """Analyze changes between branches using LLM"""
        changes = self.get_branch_diff(branch_name)
        
        if not changes:
            self.logger.info("No changes found to analyze")
            return {}

        analysis_results = {}
        review_template = """
        Analyze the following code changes between branches and provide detailed feedback on:
        1. Impact Analysis:
           - What components/functionality are affected?
           - Are there potential breaking changes?
           
        2. Code Quality:
           - Code style and best practices
           - Potential bugs or issues
           - Architecture and design patterns
           
        3. Security Review:
           - Security vulnerabilities
           - Authentication/authorization concerns
           - Data handling issues
           
        4. Performance Impact:
           - Performance implications
           - Resource usage concerns
           - Scalability considerations
           
        5. Testing Requirements:
           - What types of tests are needed?
           - Edge cases to consider
           - Integration test scenarios

        Changes for file {file_path}:
        {diff_content}
        
        Please provide a structured, detailed review focusing on the most important aspects first.
        """
        
        prompt = PromptTemplate(
            input_variables=["file_path", "diff_content"],
            template=review_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)

        for file_path, change_info in changes.items():
            self.logger.info(f"Analyzing changes in {file_path}")
            
            try:
                review = chain.run(
                    file_path=file_path,
                    diff_content=change_info['content']
                )
                
                analysis_results[file_path] = {
                    'review': review,
                    'stats': change_info['stats'],
                    'status': change_info['status']
                }
            except Exception as e:
                self.logger.error(f"Error analyzing {file_path}: {e}")
                continue

        return analysis_results

    def save_analysis(self, analysis_results: Dict, output_format: str = "markdown", output_dir: str = "./analysis_results") -> str:
        """
        Save analysis results to a file in the specified format.
        
        Args:
        analysis_results (Dict): The analysis results to save.
        output_format (str): The format of the output file ("markdown" or "yaml").
        output_dir (str): The directory where the file will be saved.

        Returns:
            str: The full path to the saved file.
        """
        # Validate output format
        if output_format not in {"markdown", "yaml"}:
            raise ValueError(f"Unsupported output format: {output_format}. Use 'markdown' or 'yaml'.")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        branch_name = self.repo.active_branch.name
        # Replace any forward slashes in branch name with underscores to avoid path issues
        safe_branch_name = branch_name.replace('/', '_')
        filename = f"branch_analysis_{safe_branch_name}_{timestamp}.{output_format}"
        file_path = os.path.join(output_dir, filename)

        # Format content based on the output format
        if output_format == "markdown":
            content = self._format_markdown(analysis_results)
        else:  # output_format == "yaml"
            content = yaml.dump(analysis_results, default_flow_style=False)

        # Save content to file
        try:
            with open(file_path, 'w') as f:
                f.write(content)
            self.logger.info(f"Analysis saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save analysis: {e}")
            raise

        return file_path

    def _format_markdown(self, analysis_results: Dict) -> str:
        """Format analysis results as markdown"""
        lines = ["# Branch Code Analysis Report\n"]
        lines.append(f"- Branch: `{self.repo.active_branch.name}`")
        lines.append(f"- Base: `{self.config.get('main_branch', 'main')}`")
        lines.append(f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        for file_path, result in analysis_results.items():
            lines.append(f"## {file_path}")
            lines.append(f"**Status**: {result['status']}")
            
            # Add statistics
            lines.append("\n### Change Statistics")
            lines.append("```")
            lines.append(f"Additions: {result['stats']['additions']}")
            lines.append(f"Deletions: {result['stats']['deletions']}")
            lines.append(f"Total Changes: {result['stats']['total_changes']}")
            lines.append("```\n")
            
            # Add review
            lines.append("### Review Analysis")
            lines.append(result['review'])
            lines.append("\n---\n")

        return "\n".join(lines)

def main():
    analyzer = BranchDiffAnalyzer()
    
    # Get current branch name
    current_branch = analyzer.repo.active_branch.name
    
    # Run analysis
    print(f"Analyzing changes in branch '{current_branch}'...")
    results = analyzer.analyze_changes()
    
    if not results:
        print("No changes found to analyze")
        return
    
    # Save results
    output_file = analyzer.save_analysis(results)
    print(f"\nAnalysis completed and saved to {output_file}")

if __name__ == "__main__":
    main()