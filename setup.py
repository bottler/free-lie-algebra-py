from setuptools import setup

setup(name='free_lie_algebra',
      version="0.01",
      url='https://github.com/bottler/free-lie-algebra-py',
      author='Jeremy Reizenstein',
      license='MIT',
      zip_safe=True,
      test_suite="free_lie_algebra.TestFLA",
      install_requires=["numpy","pyparsing","sympy","six","scipy"],
      py_modules=["free_lie_algebra"]
      )


