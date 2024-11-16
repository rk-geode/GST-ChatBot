from setuptools import find_packages,setup

setup   (            

    name='mcqgen',
    version='0.1',
    author='Rajesh',
    install_requires=["openai", "langchain","streamlit","python-dotenv","pypdf2"],
    packages=find_packages()

)