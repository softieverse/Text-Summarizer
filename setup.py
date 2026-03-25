import setuptools

with open("README.md" , "r" , encoding ="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "Text-Summarizer"
AUTHOR_USER_NAME = "softieverse"
SRC_REPO = "textSummarizer"
AUTHOR_EMAIL="muskanthakurthakur587@gmail.com"

setuptools.setup(  #converts our project into package we can later use this as an pip install 
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    autor_email=AUTHOR_EMAIL,
    description="A small python package for NLP_app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker":f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
    },
    package_dir={"":"src"},
)