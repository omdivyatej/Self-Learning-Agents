from setuptools import setup, find_packages
import os
import re

# Extract version from __init__.py
with open(os.path.join('dead_simple_self_learning', '__init__.py'), 'r') as f:
    version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]*)['\"]", f.read())
    version = version_match.group(1) if version_match else '0.1.0'

# Read the contents of README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Define required dependencies
REQUIRED_PACKAGES = [
    'numpy>=1.20.0',
    'sentence-transformers>=2.2.0',
]

# Define optional dependencies
EXTRA_REQUIRES = {
    'openai': ['openai>=1.0.0'],
    'dev': [
        'pytest>=7.0.0',
        'black>=22.0.0',
        'flake8>=5.0.0',
        'mypy>=0.9.0',
    ],
    'doc': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=1.0.0',
    ]
}

setup(
    name="dead_simple_self_learning",
    version=version,
    packages=find_packages(exclude=["tests", "examples"]),
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_REQUIRES,
    author="AI Assistant",
    author_email="example@example.com",
    description="A lightweight library for LLM self-improvement through feedback",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dead_simple_self_learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/dead_simple_self_learning/issues",
        "Source": "https://github.com/yourusername/dead_simple_self_learning",
        "Documentation": "https://github.com/yourusername/dead_simple_self_learning#readme",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 4 - Beta",
    ],
    keywords="llm, embedding, self-learning, feedback, prompt-engineering, nlp",
    python_requires=">=3.7",
    include_package_data=True,
    zip_safe=False,
) 