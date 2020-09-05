from setuptools import setup

setup(
    name="cvxstoc",
    version="0.2.1",
    author="Alnur Ali",
    packages=["cvxstoc"],
    license="GPLv3",
    description="A domain-specific language for modeling stochastic convex optimization problems in Python.",
    install_requires=[
        "cvxpy >= 0.3.5",
        "pymc >= 2.3.4",
        "numpy >= 1.10",
        "scipy >= 0.16",
    ],
    use_2to3=True,
)
