from setuptools import setup, find_packages
from salure_tfx_extensions import version


with open('README.md') as f:
    long_description = f.read()

setup(
    name='salure_tfx_extensions',
    version='{}.{}.{}'.format(
        version.VERSION_MAJOR,
        version.VERSION_MINOR,
        version.VERSION_PATCH),
    description='TFX components, helper functions and pipeline definition, developed by Salure',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Salure',
    author_email='bi@salure.nl',
    license='Salure License',
    packages=find_packages(),
    package_data={'salure_tfx_extensions': ['proto/*.proto']},
    install_requires=[
        'tfx>={}'.format(version.TFX_VERSION),
        # 'tensorflow>=1.15.0',
        # 'beam-nuggets>=0.15.1,<0.16',
        'PyMySQL>=0.9.3,<0.10'
    ],
    zip_safe=False
)


