from distutils.core import setup
import os

lib_requires = ["torch", "tqdm", "scipy", "numpy"]
test_requires = lib_requires + ["pytest", "pytest-cov", "pandas"]
demos_requires = lib_requires + ["jupyter", "matplotlib"]

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r') as readme:
     long_description = readme.read()

setup(name="cardiomyocyte_emulator",
    version="0.9",    
    description="This repository demonstrates the emulator of the paper 'Neural network emulation of the human ventricular cardiomyocyte action potential: a tool for more efficient computation in pharmacological studies'",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thomgrand/cardiomyocyte_emulator",
    packages=["cardiomyocyte_emulator"],
    install_requires=lib_requires,
    python_requires='>=3.7',
     author="Grandits et al. (see associated paper)",
     author_email="tomdev@gmx.net",
     license="AGPL",     
     extras_require = {
          'tests': test_requires,
          'docs': ["sphinx", "pydata_sphinx_theme", "nbsphinx", "sphinx-gallery", "nbsphinx_link"],
          'demos': demos_requires
     }
)
