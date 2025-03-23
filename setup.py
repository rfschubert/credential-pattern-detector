from setuptools import setup, find_packages

setup(
    name="credential_detector",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "regex>=2022.1.18",
        "joblib>=1.1.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.8",
    author="RFSchubert",
    author_email="rfschubert@example.com",
    description="Detector de padrões de credenciais baseado em IA",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rfschubert/credential-pattern-detector",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
) 