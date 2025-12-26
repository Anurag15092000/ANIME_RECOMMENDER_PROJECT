from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()
    
setup(
    name="Anime_Recommender_system",
    version="1.0",
    author="Anurag Dey",
    packages=find_packages(),
    install_requires=requirements
)