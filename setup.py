from setuptools import find_namespace_packages, setup

setup(
    name="liquid-fusion",
    version="0.2.0",
    packages=find_namespace_packages(
        include=["data*", "models*", "scripts*"],
        exclude=["tests", "tests.*"],
    ),
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "transformers==4.36.0",
        "datasets>=2.15.0",
        "rouge-score>=0.1.2",
        "tqdm>=4.65.0",
    ],
    extras_require={"dev": ["pytest>=7.0.0"]},
    python_requires=">=3.10",
)