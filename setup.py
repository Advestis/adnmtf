import subprocess
from pathlib import Path
from typing import List

from setuptools import find_packages, setup


def run_cmd(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split(" ")
    return subprocess.check_output(cmd).decode(encoding="UTF-8").split("\n")


def get_greatest_version(versions: List[str]) -> str:
    g_major, g_minor = -1, -1
    for v in versions:
        if v.startswith("v"):
            major, minor = map(int, v[1:].split("."))
            if major > g_major:
                g_major = major
                g_minor = minor
            elif minor > g_minor:
                g_minor = minor
    if g_major == g_minor == "-1":
        run_cmd("git tag v0.1")
        g_major = "0"
        g_minor = "1"
    return f"v{g_major}.{g_minor}"


def get_last_tag() -> str:
    result = run_cmd("git tag -l v*")
    return get_greatest_version(result)


def get_nb_commits_until(tag: str) -> int:
    return len(run_cmd(f'git log {tag}..HEAD --oneline'))


def get_version() -> str:
    last_tag = get_last_tag()
    return f"{last_tag[1:]}.{get_nb_commits_until(last_tag)}"


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
