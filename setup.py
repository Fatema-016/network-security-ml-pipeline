from setuptools import find_packages, setup   # find_packages : It scans directories, looks for __init__.py files, and returns a list of detected Python packages for distribution packaging.
from typing import List

HYPHEN_E_DOT = "-e ."  #  it tells pip to install networksecurity package locally in editable mode, which is what setup.py handles.

def get_requirements() -> List[str]:
    """
    This function returns a list of requirements
    """
    requirement_list: List[str] = []
    try:
        with open("requirements.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != HYPHEN_E_DOT:
                    requirement_list.append(requirement)
    except FileNotFoundError:
        pass
    return requirement_list

setup(
    name="networksecurity",
    version="0.0.1",
    author="Fatema",
    author_email="fatemahab.786@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)