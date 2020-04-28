from setuptools import setup

setup(
    name="xray_classifier",
    version="0.0.1",
    packages=["xray_classifier"],
    package_data={"xray_classifier": ["models/*"]},
    entry_points={"console_scripts": ["xray = xray_classifier.main:main"]},
    install_requires=["click", "tensorflow", "pandas", "numpy", "scikit-learn"],
    author="Justin Kohl",
)
