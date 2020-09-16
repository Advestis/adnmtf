import subprocess
from pathlib import Path
from typing import List

from setuptools import find_packages, setup


def run_cmd(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split(" ")
    return subprocess.check_output(cmd).decode(encoding="UTF-8").split("\n")


def get_last_tag() -> str:
    result = [v for v in run_cmd("git tag -l v*") if not v == ""]
    if len(result) == 0:
        run_cmd("git tag v0.0.1")
    result = [v for v in run_cmd("git tag -l v*") if not v == ""]
    return result[-1]


def get_nb_commits_until(tag: str) -> int:
    return len(run_cmd(f'git log {tag}..HEAD --oneline'))


def get_version() -> str:
    last_tag = get_last_tag()
    return f"{'.'.join(last_tag.split('.')[:-1])}.{get_nb_commits_until(last_tag)}"


long_description = Path("README.md").read_text()
requirements = Path("requirements.txt").read_text().splitlines()
version = get_version()


if __name__ == "__main__":
    setup(
        name="nmtf",
        version=version,
        packages=find_packages(),
        include_package_data=True,
        long_description=long_description,
        install_requires=requirements,
        package_data={"": ["*", ".*"]},
    )
