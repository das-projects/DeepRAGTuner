#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="DeepRAGTuner",
    version="0.0.1",
    description="End to End RAG fine tuning",
    author="Arijit Das",
    author_email="arijit.das@selfsupervised.de",
    url="https://github.com/das-projects/DeepRAGTuner",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
