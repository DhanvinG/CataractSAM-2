from setuptools import setup, find_packages

setup(
    name="cataractsam2",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,      # ships YAML inside cfg/
    python_requires=">=3.9",
    install_requires=[l.strip() for l in open("requirements.txt") if l.strip()],
    author="Dhanvin Ganeshkumar",
    license="MIT",
    description="Domain‑adapted SAM‑2 for cataract surgery videos",
)
