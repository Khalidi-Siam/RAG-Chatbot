from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function reads the requirements from a file and returns them as a list.
    '''
    with open(file_path, 'r') as file:
        requirements = file.read().splitlines()
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


__version__ = '0.0.1'
REPO_NAME = "RAG-Chatbot"
AUTHOR_USER_NAME = "khalidi-siam"
SRC_REPO = "RAG-Chatbot"
AUTHOR_EMAIL = "siam074@yahoo.com"


setup(
    name='RAG-Chatbot',
    version='0.0.1',
    author="Khalidi Siam",
    author_email="siam074@yahoo.com",
    description="A Retrieval-Augmented Generation (RAG) Chatbot that combines retrieval-based and generative approaches to provide accurate and contextually relevant responses.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=get_requirements('requirements.txt'),
)