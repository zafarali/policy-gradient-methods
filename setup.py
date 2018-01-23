from setuptools import setup, find_packages
from pg_methods import __version__

setup(
    name='pg_methods',
    version=__version__,
    description='Policy Gradient methods implemented in pytorch',
    long_description=open('README.md', encoding='utf-8').read(),
    url='https://github.com/zafarali/policy-gradient-methods',
    author='Zafarali Ahmed',
    author_email='zafarali.ahmed@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha'
    ],
    python_requires='>=3.5'
)