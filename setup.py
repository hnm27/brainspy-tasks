from setuptools import setup, find_packages

setup(
    name="brainspy-tasks",
    version="1.0.0",
    description="Benchmark tests and tasks for studying the capacity of boron-doped silicon devices and their surrogate models.",
    url="https://github.com/BraiNEdarwin/brainspy-tasks",
    author="Unai Alegre-Ibarra et al.",
    author_email="u.alegre@utwente.nl",
    license="GPL-3.0",
    packages=find_packages(),
    install_requires=["brainspy",
    "brainspy-smg",
    "notebook",
    "tensorboard" ],
    zip_safe=False,
)
