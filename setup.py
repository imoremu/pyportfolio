import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyportfolio",
    version="0.1.0",
    author="Iván Moreno",
    author_email="ivan.moreno@nereodata.com",
    description="Una librería para la gestión y cálculo de carteras de inversión.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/imoremu/pyportfolio",     
    packages=setuptools.find_packages(include=["pyportfolio*"]),    
    include_package_data=True,    
    classifiers=[        
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires='>=3.8',
    install_requires=[        
        "pandas>=1.0",
        "pydatastudio", 
    ],)