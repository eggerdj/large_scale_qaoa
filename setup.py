"""Setup for large-scale QAOA."""

import os
import setuptools

long_description = """repository for large-scale QAOA."""

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

VERSION_PATH = os.path.join(os.path.dirname(__file__), "large_scale_qaoa", "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()

setuptools.setup(
    name="large_scale_qaoa",
    version=VERSION,
    description="Code to run large-scale QAOA experiments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eggerdj/large_scale_qaoa",
    author="Stefan H. Sack and Daniel J. Egger",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit qaoa",
    packages=setuptools.find_packages(
        include=["large_scale_qaoa", "large_scale_qaoa.*"]
    ),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    python_requires=">=3.6",
    zip_safe=False,
)
