from setuptools import setup, find_packages

# IMPORTANT: ALSO UPDATE IN 'VERSION' FILE FOR CI DOCKER BUILD
VERSION_MAJOR = '0'
VERSION_MINOR = '2'
VERSION_PATCH = '15'


with open('README.md') as f:
    long_description = f.read()

setup(
    name='salure_tfx_extensions',
    version='{}.{}.{}'.format(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH),
    python_requires='>=3.6',
    description='TFX components, helper functions and pipeline definition, developed by Salure',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Salure',
    author_email='bi@salure.nl',
    license='Salure License',
    packages=find_packages(),
    package_data={'salure_tfx_extensions': ['proto/*.proto']},
    install_requires=[
        'beam-nuggets>=0.15.1,<0.16',
        'tfx>=0.27.0,<0.28.0',
        'pandas>=1,<=1.35',
        'kfp>=0.4,<0.5',
        'PyMySQL>=1,<2',
        'pyparsing==2.4.2',
        'cryptography'
    ],
    zip_safe=False
)
