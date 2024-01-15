from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = "A package for using Openai in serverless environment"
LONG_DESCRIPTION = 'A package for using Openai with scraping and etc. in serverless application such as AWS Lambda and GCP Cloud Function'

# Setting up
setup(
    name="serverless_openai",
    version=VERSION,
    author="Jayr Castro",
    author_email="jayrcastro.py@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['requests', 'opencv-python-headless', 'beautifulsoup4', 'numpy', 'pydantic'],
    keywords=['serverless', 'openai', 'aws lambda', 'cloud functions', 'openai API'],
    classifiers=[
        "Development Status :: 2 - Developing",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)