from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="inference_time_scaling",
    version="0.1.0",
    author="Kai Xu and the Red Hat AI Innovation Team",
    author_email="xuk@redhat.com",
    description="A Python library for inference-time scaling LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Red-Hat-AI-Innovation-Team/inference_time_scaling",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "typing-extensions>=4.0.0",  # for Protocol support in older Python versions
    ],
) 