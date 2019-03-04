from setuptools import setup
import subprocess


def get_development_version():
    git_output = subprocess.run(['git', 'rev-list', '--count', 'master'], stdout=subprocess.PIPE)
    version = '0.%s' % git_output.stdout.decode('utf-8').strip()
    return version


def get_requirements():
    with open('requirements.txt') as f:
        requirements = [p.strip().split('=')[0] for p in f.readlines() if p[0] != '-']
    return requirements


setup(
    name='ou_noise',
    version=get_development_version(),
    author='Julian Wergieluk',
    author_email='julian@wergieluk.com',
    packages=['ou_noise', ],
    url='',
    install_requires=get_requirements(),
    description='Stochastic processes simulators',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
