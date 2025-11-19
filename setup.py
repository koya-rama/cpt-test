"""
Setup script for Nemotron CPT Pipeline
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nemotron-cpt",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Continuous Pre-Training Pipeline for NVIDIA Nemotron Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nemotron-cpt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "nemotron-cpt=src.training.train_cpt:main",
        ],
    },
)
