from setuptools import setup, find_packages

setup(
    name="zign",
    version="0.0.6",
    url="https://github.com/lvancer/zign",
    author="lvancer",
    author_email="lin029011@163.com",
    license="MIT",
    description="",
    packages=find_packages(exclude=("examples", "examples.*", "tests", "tests.*")),
)