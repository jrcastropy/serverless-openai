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
    install_requires=[
        "annotated-types==0.6.0",
        "beautifulsoup4==4.12.2",
        "certifi==2023.11.17",
        "charset-normalizer==3.3.2",
        "idna==3.6",
        "numpy==1.24.4",
        "opencv-python-headless==4.9.0.80",
        "pydantic==2.5.3",
        "pydantic-core==2.14.6",
        "requests==2.31.0",
        "soupsieve==2.5",
        "typing-extensions==4.9.0",
        "urllib3==2.1.0"
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