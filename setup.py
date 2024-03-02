from typing import List

import setuptools


def parse_requirements_txt(filename: str) -> List[str]:
    with open(filename) as fd:
        return list(filter(lambda line: bool(line.strip()), fd.read().splitlines()))


setuptools.setup(
    name="ctrlNetFun",  # Replace with your own username
    version="0.0.1",
    author="Victor Kulikov",
    description="ControlNet package",
    url="",
    packages=['src', 'src/controlnet'],
    include_package_data=False,
    python_requires=">=3.6, <4",
    zip_safe=False,
    entry_points={"console_scripts": [
        "ctrlnet = src.inference:main",
    ]},
    install_requires=parse_requirements_txt("requirements.txt"),
)
