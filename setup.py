from setuptools import setup, find_packages

setup(
    name="world_model_rl",
    version="0.1.0",
    description="World-model-based reinforcement learning for puzzle environments",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "gymnasium>=0.29",
        "pygame>=2.5",
        "numpy>=1.24",
        "matplotlib>=3.7",
        "streamlit>=1.30",
        "wandb>=0.16",
        "pyyaml>=6.0",
        "stable-baselines3>=2.2",
        "plotly>=5.18",
        "imageio>=2.33",
    ],
)
