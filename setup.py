from setuptools import setup
from setuptools import find_packages


def convert_md_to_rst(path):
    try:
        from pypandoc import convert_file
    except ImportError:
        print("warning: pypandoc module not found, could not convert Markdown to RST")
        return open(path, 'r').read()

    return convert_file(path, 'rst')


numerauto_version = 'dev'


classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GPL License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering"]


if __name__ == "__main__":
    setup(
        name="numerauto",
        version=numerauto_version,
        maintainer="The Brain",
        maintainer_email="thebrainz1985@gmail.com",
        description="Daemon to facilitate automatically competing in the Numerai machine learning competition",
        long_description=convert_md_to_rst('README.md'),
        url='https://github.com/thebrain85/numerauto',
        platforms="OS Independent",
        classifiers=classifiers,
        license='GPL License',
        package_data={'numerauto': ['LICENSE', 'README.md']},
        packages=find_packages(exclude=['tests']),
        python_requires='>=3',
        install_requires=["requests", "pytz", "python-dateutil", "pandas", "numerapi"]
    )
