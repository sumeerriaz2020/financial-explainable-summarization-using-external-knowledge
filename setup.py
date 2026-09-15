"""
Financial Explainable Summarization Setup
==========================================

Setup configuration for pip installation.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f 
                if line.strip() and not line.startswith('#')]

# Core requirements
install_requires = read_requirements('requirements.txt')

# Development requirements
dev_requires = [
    'pytest>=7.4.0',
    'pytest-cov>=4.1.0',
    'black>=23.7.0',
    'flake8>=6.1.0',
    'mypy>=1.5.0',
    'sphinx>=7.1.0',
    'sphinx-rtd-theme>=1.3.0',
]

setup(
    name="fin-explainable",
    version="1.0.0",
    author="Sumeer Riaz, Dr. M. Bilal Bashir, Syed Ali Hassan Naqvi",
    author_email="sumeer33885@iqraisb.edu.pk",
    description="Hybrid Neural-Symbolic Framework for Explainable Financial Document Summarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sumeerriaz2020/financial-explainable-summarization-using-external-knowledge",
    project_urls={
        "Bug Tracker": "https://github.com/sumeerriaz2020/financial-explainable-summarization-using-external-knowledge/issues",
        "Documentation": "https://github.com/sumeerriaz2020/financial-explainable-summarization-using-external-knowledge/blob/main/README.md",
        "Source Code": "https://github.com/sumeerriaz2020/financial-explainable-summarization-using-external-knowledge",
    },
    # Flat top-level packages matching the on-disk layout. This project is intended
    # to be used from a clone rather than installed from an index.
    packages=find_packages(exclude=['test', 'test.*', 'docs', 'documentation']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business :: Financial",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        'dev': dev_requires,
        'all': dev_requires,
    },
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.yml', '*.owl', '*.md'],
    },
    # No console_scripts are declared; a CLI entry point is listed under
    # "Known gaps" in README.md.
    keywords=[
        'financial-nlp',
        'explainable-ai',
        'text-summarization',
        'knowledge-graph',
        'hybrid-neural-symbolic',
        'fibo',
        'transformers',
    ],
    zip_safe=False,
)
