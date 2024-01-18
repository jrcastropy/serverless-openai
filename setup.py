from setuptools import setup, find_packages

VERSION = '1.2.3'
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
    install_requires=[
        "pydantic==2.5.3",
        "beautifulsoup4==4.12.2",
        "opencv-python-headless==4.9.0.80",
        "requests==2.28.2",
        "typing-extensions==4.9.0",
        "tiktoken==0.5.2"
    ],
    keywords=['serverless', 'openai', 'aws lambda', 'cloud functions', 'openai API'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)