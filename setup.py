from setuptools import setup, find_packages

setup(
    name="brainspy-tasks",
    version="0.8.1",
    description="Benchmark tests and tasks for studying the capacity of boron-doped silicon devices and their surrogate models.",
    url="https://github.com/BraiNEdarwin/brainspy-tasks",
    author="This has adopted part of the BRAINS skynet repository code, which has been cleaned and refactored. Some new tests have been created. The maintainers of the code are Hans-Christian Ruiz Euler and Unai Alegre Ibarra.",
    author_email="u.alegre@utwente.nl",
    license="GPL-3.0",
    packages=find_packages(),
    install_requires=["matplotlib",
"tensorboard" ],
    zip_safe=False,
)
