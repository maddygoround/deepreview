from setuptools import setup, find_packages

setup(
    name="deepreview",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gitpython",
        "langchain",
        "pyyaml",
        "ollama"
    ],
    entry_points={
        'console_scripts': [
            'deepreview=deepreview.deepreview:main',
        ],
    },
)