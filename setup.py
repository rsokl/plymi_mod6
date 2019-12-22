from setuptools import find_packages, setup

setup(
    name="plymi_mod_6",
    packages=find_packages(exclude=["tests", "tests.*"]),
    version="1.0.0",
    author="A Fastidious PLYMI Reader",
    author_email="plymi.rocks@plymi.com",
    description="A template Python package for learning about testing",
    install_requires=["numpy >= 1.10.0"],
    tests_require=["pytest", "hypothesis"],
    python_requires=">=3.6",
)
