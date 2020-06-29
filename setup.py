from setuptools import setup, find_packages

# IMPORTANT: ALSO UPDATE IN 'VERSION' FILE FOR CI DOCKER BUILD
VERSION_MAJOR = '0'
VERSION_MINOR = '0'
VERSION_PATCH = '52'


with open('README.md') as f:
    long_description = f.read()

setup(
    name='salure_tfx_extensions',
    version='{}.{}.{}'.format(
        VERSION_MAJOR,
        VERSION_MINOR,
        VERSION_PATCH),
    description='TFX components, helper functions and pipeline definition, developed by Salure',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Salure',
    author_email='bi@salure.nl',
    license='Salure License',
    packages=find_packages(),
    package_data={'salure_tfx_extensions': ['proto/*.proto']},
    install_requires=[
        # 'tfx>={}'.format(version.TFX_VERSION),
        'tfx>=0.22.0,<0.23.0',
        # 'tensorflow>=1.15.0',
        # 'beam-nuggets>=0.15.1,<0.16',
        'PyMySQL>=0.9.3,<0.10'
    ],
    zip_safe=False
)


