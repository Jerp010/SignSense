"""
SignSense - ASL Sign Language Learning Application

Setup script for backward compatibility with older Python packaging tools.
For modern packaging, use pyproject.toml.
"""

from setuptools import setup, find_packages

setup(
    name="signsense",
    version="1.0.0",
    description="SignSense - ASL Sign Language Learning Application",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="SignSense Team",
    python_requires=">=3.9,<3.13",
    packages=find_packages(include=["signsense", "signsense.*"]),
    package_data={
        "signsense": [
            "assets/models/*.task",
            "config/*.yaml",
            "ml/config/*.yaml",
        ],
    },
    install_requires=[
        "mediapipe>=0.10.31",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "pillow>=9.0.0",
        "pandas>=1.5.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.2.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "flake8>=6.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "signsense=signsense.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Multimedia :: Video :: Capture",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="sign-language asl machine-learning opencv mediapipe",
)
