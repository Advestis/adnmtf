import subprocess
from pathlib import Path
from typing import List

from setuptools import find_packages, setup


def run_cmd(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split(" ")
    return subprocess.check_output(cmd).decode(encoding="UTF-8").split("\n")


def get_greatest_version(versions: List[str]) -> str:
    versions = [list(map(int, v[1:].split("."))) for v in versions]
    greatest = None
    for v in versions:
        if greatest is None:
            greatest = v
        else:
            lower = False
            for i in range(len(v)):
                if len(greatest) < i + 1:
                    greatest = v
                    break
                if v[i] > greatest[i]:
                    greatest = v
                    break
                if v[i] < greatest[i]:
                    lower = True
                    break
            if not lower:
                greatest = v
    return f"v{'.'.join([str(s) for s in greatest])}"


def get_last_tag() -> str:
    result = [v for v in run_cmd("git tag -l v*") if not v == ""]
    if len(result) == 0:
        run_cmd("git tag v0.1")
    result = [v for v in run_cmd("git tag -l v*") if not v == "" and v.startswith("v")]
    return get_greatest_version(result)


def get_nb_commits_until(tag: str) -> int:
    return len(run_cmd(f'git log {tag}..HEAD --oneline'))


def get_version() -> str:
    last_tag = get_last_tag()
    return f"{'.'.join(last_tag.split('.'))}.{get_nb_commits_until(last_tag)}"


try:
    long_description = Path("README.md").read_text()
except UnicodeDecodeError:
    with open("README.md", "rb") as ifile:
        lines = [line.decode("utf-8") for line in ifile.readlines()]
        long_description = "".join(lines)

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
