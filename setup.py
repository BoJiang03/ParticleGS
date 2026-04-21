from setuptools import setup, find_packages

setup(
    name="particlegs",
    version="0.1.0",
    description="ParticleGS: Visualization-Aware Gaussian Splatting for Scientific Particle Data",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "plyfile",
        "tqdm",
        "Pillow",
        "opencv-python",
        "scipy",
        "scikit-image",
    ],
)
