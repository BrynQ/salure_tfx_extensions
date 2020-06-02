from setuptools import setup, find_packages

setup(
    name='salure_tfx_extensions',
    version='0.0.34',
    description='TFX components, helper functions and pipeline definition, developed by Salure',
    author='Salure',
    author_email='bi@salure.nl',
    license='Salure License',
    packages=find_packages(),
    package_data={'salure_tfx_extensions': ['proto/*.proto']},
    install_requires=[
        'tfx>=0.14.0rc1,<0.15',
        'tensorflow==1.14.0',
        # 'beam-nuggets>=0.15.1,<0.16',
        'PyMySQL>=0.9.3,<0.10'
    ],
    zip_safe=False
)
