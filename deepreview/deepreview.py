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
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from langchain.globals import set_llm_cache, get_llm_cache
from langchain.cache import InMemoryCache

class BranchDiffAnalyzer:
    def __init__(self, config_path: str = "config.yaml"):
        set_llm_cache(InMemoryCache())
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
        
        # Initialize thread-local storage for LLM instances
        self.thread_local = threading.local()
        
        # Get max workers from config or default to CPU count
        self.max_workers = self.config.get("max_workers", os.cpu_count() or 4)
        
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

    def get_llm(self):
        """Get or create thread-local LLM instance"""
        if not hasattr(self.thread_local, "llm"):
            self.thread_local.llm = Ollama(model=self.config.get("llm_model", "codellama"))
        return self.thread_local.llm

    def analyze_file(self, file_path: str, change_info: Dict) -> Dict:
        """Analyze a single file's changes using LLM"""
        self.logger.info(f"Analyzing changes in {file_path}")
        
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
        
        try:
            # Get thread-local LLM instance
            llm = self.get_llm()
            chain = LLMChain(llm=llm, prompt=prompt)
            
            review = chain.run(
                file_path=file_path,
                diff_content=change_info['content']
            )
            
            return {
                'review': review,
                'stats': change_info['stats'],
                'status': change_info['status']
            }
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            return None

    def analyze_changes(self, branch_name: Optional[str] = None) -> Dict:
        """Analyze changes between branches using LLM with parallel processing"""
        changes = self.get_branch_diff(branch_name)
        
        if not changes:
            self.logger.info("No changes found to analyze")
            return {}

        analysis_results = {}
        
        # Create a thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all file analysis tasks
            future_to_file = {
                executor.submit(self.analyze_file, file_path, change_info): file_path
                for file_path, change_info in changes.items()
            }
            
            # Process completed analyses as they finish
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        analysis_results[file_path] = result
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
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
    
    def setup_qa_llm(self):
        """Setup a separate LLM for QA interactions"""
        qa_model = self.config.get("qa_model", "gpt4all")
        return Ollama(model=qa_model)

    def interactive_qa(self, analysis_results: Dict):
        """
        Start an interactive QA session about the code changes.
        Continues until user types 'exit' or sends keyboard interrupt.
        
        Args:
            analysis_results (Dict): The analysis results to reference during QA
        """
        qa_llm = self.setup_qa_llm()
        
        # Create a context summary for the QA session
        context = self._create_qa_context(analysis_results)
        
        qa_template = """
        You are a helpful AI assistant specialized in code review discussions.
        You have access to the following code change analysis:
        
        {context}
        
        Based on this context, please answer the following question:
        {question}
        
        Provide clear, concise answers and be ready to go into technical details when needed.
        If you're unsure about something, say so rather than making assumptions.
        """
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=qa_template
        )
        
        chain = LLMChain(llm=qa_llm, prompt=prompt)
        
        print("\nStarting QA session about code changes.")
        print("Type 'exit' to end the session.")
        print("-" * 50)
        
        try:
            while True:
                question = input("\nYour question: ").strip()
                
                if question.lower() == 'exit':
                    print("\nEnding QA session.")
                    break
                    
                if not question:
                    continue
                    
                try:
                    response = chain.run(
                        context=context,
                        question=question
                    )
                    print("\nAnswer:", response)
                    
                except Exception as e:
                    self.logger.error(f"Error in QA response: {e}")
                    print("\nSorry, I encountered an error processing your question. Please try again.")
                    
        except KeyboardInterrupt:
            print("\n\nQA session terminated by user.")
            return

    def _create_qa_context(self, analysis_results: Dict) -> str:
        """
        Create a condensed context from analysis results for the QA chain.
        
        Args:
            analysis_results (Dict): The analysis results to summarize
            
        Returns:
            str: A condensed context string
        """
        context_parts = []
        
        for file_path, result in analysis_results.items():
            context_parts.append(f"File: {file_path}")
            context_parts.append(f"Status: {result['status']}")
            context_parts.append("Changes:")
            context_parts.append(f"- Additions: {result['stats']['additions']}")
            context_parts.append(f"- Deletions: {result['stats']['deletions']}")
            context_parts.append(f"- Total Changes: {result['stats']['total_changes']}")
            context_parts.append("\nAnalysis Summary:")
            context_parts.append(result['review'])
            context_parts.append("-" * 40 + "\n")
        
        return "\n".join(context_parts)
    
    def _parse_markdown_to_dict(self, markdown_content: str) -> Dict:
        """
        Parse markdown content back into a dictionary format similar to analysis results.
        
        Args:
            markdown_content (str): The markdown content to parse
            
        Returns:
            Dict: Parsed content in the same format as analysis results
        """
        results = {}
        current_file = None
        current_section = None
        
        # Split content into lines
        lines = markdown_content.split('\n')
        
        for line in lines:
            # File header (## filename)
            if line.startswith('## '):
                current_file = line[3:].strip()
                results[current_file] = {
                    'status': '',
                    'stats': {'additions': 0, 'deletions': 0, 'total_changes': 0},
                    'review': ''
                }
                continue
                
            if not current_file:
                continue
                
            # Status line
            if line.startswith('**Status**:'):
                results[current_file]['status'] = line.split(':')[1].strip()
                continue
                
            # Stats section
            if 'Change Statistics' in line:
                current_section = 'stats'
                continue
                
            # Review section
            if 'Review Analysis' in line:
                current_section = 'review'
                continue
                
            # Parse stats
            if current_section == 'stats':
                if line.startswith('Additions:'):
                    results[current_file]['stats']['additions'] = int(line.split(':')[1].strip())
                elif line.startswith('Deletions:'):
                    results[current_file]['stats']['deletions'] = int(line.split(':')[1].strip())
                elif line.startswith('Total Changes:'):
                    results[current_file]['stats']['total_changes'] = int(line.split(':')[1].strip())
                    
            # Collect review content
            if current_section == 'review' and line and not line.startswith('###'):
                results[current_file]['review'] = results[current_file]['review'] + line + '\n'
        
        return results

    def load_analysis_file(self, file_path: str) -> Dict:
        """
        Load and parse analysis results from a file.
        
        Args:
            file_path (str): Path to the analysis file (yaml or markdown)
            
        Returns:
            Dict: Analysis results in standardized format
        """
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.yaml'):
                    return yaml.safe_load(f)
                elif file_path.endswith('.md') or file_path.endswith('.markdown'):
                    content = f.read()
                    return self._parse_markdown_to_dict(content)
                else:
                    raise ValueError("Unsupported file format. Use 'markdown' or 'yaml'.")
        except Exception as e:
            self.logger.error(f"Error loading analysis file: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Branch Diff Analyzer")
    parser.add_argument('--qa', action='store_true', help="Include QA chain in the analysis")
    parser.add_argument('--output', type=str, help="Specify the output file path for the analysis results")
    parser.add_argument('--workers', type=int, help="Number of worker threads for parallel processing")
    args = parser.parse_args()

    analyzer = BranchDiffAnalyzer()
    
    # Override max_workers if specified in command line
    if args.workers:
        analyzer.max_workers = args.workers

    # Initialize results
    if args.output:
        try:
            results = analyzer.load_analysis_file(args.output)
        except Exception as e:
            print(f"Error loading analysis file: {e}")
            return
    else:
        current_branch = analyzer.repo.active_branch.name
        print(f"Analyzing changes in branch '{current_branch}' using {analyzer.max_workers} workers...")
        results = analyzer.analyze_changes()
        
        if not results:
            print("No changes found to analyze")
            return
        
        output_file = analyzer.save_analysis(results)
        print(f"\nAnalysis completed and saved to {output_file}")

    if args.qa:
        analyzer.interactive_qa(results)

if __name__ == "__main__":
    main()