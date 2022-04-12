import os
import numpy as np
from setuptools import Extension, setup, find_packages
from setuptools.command.install import install

libraries = []
if os.name == 'posix':
    libraries.append('m')

extensions = []
for name in ['_tree', '_splitter', '_criterion', '_utils']:
    extensions.append(Extension(
        'actableai.third_parties.skgarden.mondrian.tree.{}'.format(name),
        sources=['actableai/third_parties/skgarden/mondrian/tree/{}.pyx'.format(name)],
        include_dirs=[np.get_include()],
        libraries=libraries,
        extra_compile_args=['-O3'],
    ))


class DownloadNLTK(install):
    def run(self):
        super().do_egg_install()
        import nltk
        nltk.download('stopwords')


setup(
    version="0.1",
    name="actableai",
    packages=find_packages(where="."),
    install_requires=[
        # actable
        "autogluon.core @ file://./third_parties/autogluon/core",
        "autogluon.features @ file://./third_parties/autogluon/features",
        "autogluon.tabular @ file://./third_parties/autogluon/tabular",
        "autogluon.mxnet @ file://./third_parties/autogluon/mxnet",
        "autogluon.text @ file://./third_parties/autogluon/text",
        "pandasql>=0.7.3",
        "fastai>=2.3.1",

        # gluonts
        "matplotlib>=3.0",
        "pydantic>=1.8",
        "ujson>=1.35",

        # pts
        "scipy>=1.3.3,<1.7.0",
        "python-rapidjson>=1.0",
        "torch>=1.7.1",

        "click==7.1.1",
        "scikit-learn>=0.24.2,<0.25",
        "keras<2.5.0",
        "joblib>=0.16.0",
        "visions>=0.6.4",
        "ray>=1.11.0",
        "hyperopt>=0.2.4",
        "prophet>=1.0.1",
        "rpy2>=2.9.*,<3.*",
        "alibi>=0.5.7,<=0.5.9",
        "tensorflow>=2.4.1,<2.5.*",
        "shap<=0.39.0",
        "sqlalchemy>=1.3.5, <2.0",
        "redis>=3.5.3",
        "filelock>=3.0.12",

        # causal
        "econml>=0.11.1",
        "dowhy>=0.6",
        "networkx>=2.5.1",

        "nltk"
    ],
    setup_requires=[
        "cython>=0.23",
        "setuptools>=18.0",
        "numpy>=1.19.0",
        "pandas>=1.1.4",
        "LunarCalendar>=0.0.9",
        "holidays>=0.10.3",
        "pystan>=2.19.1.1",
        "tqdm>=4.23",
        "wheel",
        "nltk"
    ],
    include_package_data=True,
    ext_modules=extensions,
    cmdclass={
        DownloadNLTK.__name__: DownloadNLTK
    }
)
