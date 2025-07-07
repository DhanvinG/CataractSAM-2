from setuptools import setup, find_packages


def _read_requirements():
    """Read requirements.txt, ignoring editable installs."""
    with open("requirements.txt") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("-e") and not line.startswith("--editable"):
                yield line

setup(
    name="cataractsam2",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,      # ships YAML inside cfg/
    package_data={"cataractsam2": ["cfg/*.yaml"]},
    python_requires=">=3.10",
    install_requires=list(_read_requirements()),
    entry_points={
    "hydra.searchpath": [
      # point to the module path and class name you just created
      "cataractsam2_cfg=cataractsam2.search_path_plugin:CataractSAM2SearchPathPlugin"
    ]
  },
    author="Dhanvin Ganeshkumar",
    license="Apache-2.0",
    description="Domain‑adapted SAM‑2 for cataract surgery videos",
)
