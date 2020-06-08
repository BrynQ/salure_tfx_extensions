from setuptools import setup, find_packages


def read_key_value_pairs_from_file(filename):
    result = {}
    with open(filename) as f:
        for line in f:
            if line[0] != '#':
                key, _, value = line.partition('=')
                result[key.strip()] = value.strip()
    return result


version_variables = read_key_value_pairs_from_file('VERSION')

with open('README.md') as f:
    long_description = f.read()

setup(
    name='salure_tfx_extensions',
    version='{}.{}.{}'.format(
        version_variables["VERSION_MAJOR"],
        version_variables["VERSION_MINOR"],
        version_variables["VERSION_HOTFIX"]),
    description='TFX components, helper functions and pipeline definition, developed by Salure',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Salure',
    author_email='bi@salure.nl',
    license='Salure License',
    packages=find_packages(),
    package_data={'salure_tfx_extensions': ['proto/*.proto']},
    install_requires=[
        'tfx>={}'.format(version_variables["TFX_VERSION"]),
        # 'tensorflow>=1.15.0',
        # 'beam-nuggets>=0.15.1,<0.16',
        'PyMySQL>=0.9.3,<0.10'
    ],
    zip_safe=False
)


