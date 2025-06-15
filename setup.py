from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Reinforcement learning with DQN in a warehouse environment',
    author='Nikit',
    license='MIT',
    install_requires=[
        'torch>=2.0.0',
        'stable-baselines3[extra]>=2.0.0',
        'numpy',
        'pandas',
        'gymnasium',
        'tensorboard',
    ],
)
