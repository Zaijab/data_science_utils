from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="your-project-name",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.9",
    author="Your Name",
    author_email="your.email@example.com",
    description="Brief description of your project",
)
