from setuptools import setup
from setuptools import find_packages


with open('README.md') as f:
    long_description = f.read()

numerauto_version = '0.1.1dev'


classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
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
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/thebrain85/numerauto',
        platforms="OS Independent",
        classifiers=classifiers,
        license='GNU General Public License v3',
        package_data={'numerauto': ['LICENSE', 'README.md']},
        packages=find_packages(),
        python_requires='>=3',
        install_requires=["requests", "pytz", "python-dateutil", "pandas", "numerapi"]
    )
