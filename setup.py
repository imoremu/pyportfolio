import setuptools

# Lee el contenido de tu README.md para la descripción larga
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
    url="https://github.com/imoremu/pyportfolio", # Opcional: URL del repositorio
    # Busca automáticamente los paquetes dentro del directorio 'pyportfolio'
    # Asegúrate de que tu código fuente principal esté en una carpeta llamada 'pyportfolio'
    # y que contenga un archivo __init__.py (puede estar vacío)
    packages=setuptools.find_packages(where="."), # Asume que setup.py está en la raíz y el paquete 'pyportfolio' también
    classifiers=[
        # Clasificadores para PyPI: https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Elige tu licencia
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires='>=3.8',
    install_requires=[        
        "pandas>=1.0",        
    ],)