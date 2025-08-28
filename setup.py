"""
Setup script for Differentiable Model Predictive Path Integral (diff-mppi) package.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements_path = os.path.join(this_directory, 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="diff-mppi",
    version="1.0.0",
    author="diff-mppi Contributors",
    author_email="your.email@example.com",
    description="Differentiable Model Predictive Path Integral Control Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kisisjrlly/diff-mppi",
    packages=find_packages(exclude=['examples', 'tests', 'docs']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.910",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "seaborn>=0.11.0",
            "tqdm>=4.60.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "diff-mppi-demo=diff_mppi.cli:main",
        ],
    },
    keywords=[
        "model predictive control",
        "path integral",
        "reinforcement learning",
        "optimal control",
        "pytorch",
        "differentiable programming",
        "mppi",
        "neural networks",
    ],
    project_urls={
        "Bug Reports": "https://github.com/kisisjrlly/diff-mppi/issues",
        "Source": "https://github.com/kisisjrlly/diff-mppi",
        "Documentation": "https://github.com/kisisjrlly/diff-mppi/blob/main/README.md",
    },
    include_package_data=True,
    package_data={
        "diff_mppi": ["*.txt", "*.md"],
    },
    zip_safe=False,
)
