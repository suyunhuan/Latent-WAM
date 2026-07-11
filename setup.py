import os

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

requirements = []
for requirements_file in ("requirements.txt", "requirements-latent-wam.txt"):
    with open(requirements_file) as f:
        requirements.extend(f.read().splitlines())

# Installs
setuptools.setup(
    name="latent-wam",
    version="0.1.0",
    author="Latent WAM contributors",
    description="Latent WAM research code built on NAVSIM",
    url="https://github.com/Jinhsi555/latent-wam",
    python_requires=">=3.10,<3.11",
    # Distribution metadata covers only the Apache-2.0 NAVSIM-derived Python
    # package. The separately licensed WorldMirror submodule is not included.
    packages=setuptools.find_packages(script_folder, include=("navsim", "navsim.*")),
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
    ],
    license="apache-2.0",
    install_requires=requirements,
)
