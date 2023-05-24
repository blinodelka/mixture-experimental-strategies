# AutoRA Template

## Quickstart Guide

### Create GitHub Repository

You should create a GitHub repository from the root folder of this project:
- Create a new repository on GitHub (a sensible name for the repository would be autora-experimentalist-sampler-mixture_experimentalist)
- Follow the guide under `…or push an existing repository from the command line
` 

### Virtual Environment 
Install this in an environment using your chosen package manager. In this example we are using virtualenv

Install:
- python (3.8 or greater): https://www.python.org/downloads/
- virtualenv: https://virtualenv.pypa.io/en/latest/installation.html

Create a new virtual environment:
```shell
virtualenv venv
```
*Note: You want to ensure that the python version matches that of autora. If necessary 
you can specify the respective python version directly, e.g., ``virtualenv venv --python=python3.9``*

Activate it:
```shell
source venv/bin/activate
```

### Install Dev Dependencies

Use `pip install` to install the current project (`"."`) in editable mode (`-e`) with dev-dependencies (`[dev]`):
```shell
pip install -e ".[dev]"
```

*Note: You may install new dependencies via ``pip install packagename`` inside your virtual environment. If those
dependencies are vital to your package, you will have to add them to the ``pyproject.toml`` (see Step 6 of the 
Contribution Guide).*

## Contribution Guide

### Experimentalist Sampler
A method that identifies novel experiment conditions $X'$ that yield scientific merit.
*Sampler* select from an existing pool of experiment conditions $X$).<br>
*Example: The [Novelty Sampler](https://github.com/AutoResearch/autora-novelty-sampler) selects novel experiment 
conditions $X'$ with respect to a pairwise distance metric applied to existing experiment conditions $X$.*
### Step 1: Implement Your Code

You may now add your code to `src/autora/experimentalist/sampler/mixture_experimentalist/__init__.py` file. You may 
also add additional files in this folder. Just make sure to import the core function or class of your feature
in the `__init__.py` if it is implemented elswhere. 

### Step 2 (Optional): Add Tests

It is highly encouraged to add unit tests to ensure your code is working as intended. These can be [doctests](https://docs.python.org/3/library/doctest.html) or test cases in `tests/test_mixture_experimentalist.py`.

*Note: Tests are required if you wish that your feature becomes part of the main 
[autora](https://github.com/AutoResearch/autora) package. However, regardless of whether you choose to implement tests, 
you will still be able to install your package separately, in addition to autora.* 

### Step 3 (Optional): Add Documentation

It is highly encouraged that you add documentation of your package in your `docs/index.md`. You can also add new pages 
in the `docs` folder. Update the `mkdocs.yml` file to reflect structure of the documentation. For example, you can add 
new pages or delete pages that you deleted from the `docs` folder.

*Note: Documentation is required if you wish that your feature becomes part of the main 
[autora](https://github.com/AutoResearch/autora) package. However, regardless of whether you choose to write
documentation, you will still be able to install your package separately, in addition to autora.*

### Step 4: Add Dependencies

In pyproject.toml add the new dependencies under `dependencies`

Install the added dependencies
```shell
pip install -e ".[dev]"
```

### Step 5: Publish Your Package

Once your project is implemented, you may publish it as subpackage of AutoRA. If you have not thoroughly vetted your project or would otherwise like to refine it further, you may 
nervous about the state of your package–you will be able to publish it as a pre-release, indicating to users that
the package is still in progress.

#### Step 5.1: Update Metadata

To begin publishing your package, update the metadata under `project` in the pyproject.toml file to include 
- name
- description
- author-name
- author-email
Also, update the URL for the repository under `project.urls`.

#### Step 5.2 Publish via GitHub Actions

To automate the publishing process for your package, you can use a GitHub action:
- Add a github secret in your repository named PYPI_API_TOKEN, that contains a PyPI token of your account
- Add the GitHub action to the `.github/workflows` directory: For example, you can use the default publishing action:
  - Navigate to the `actions` on the GitHub website of your repository.
  - Search for the `Publish Python Package` action and add it to your project
- Create a new release: Click on `create new release` on the GitHub website of your repository.
- Choose a tag (this is the version number of the release. If you didn't set up dynamic versioning it should match the version in the `pyproject.toml` file)
- Generate release notes automatically by clicking `generate release`, which adds the markdown of the merged pull requests and the contributors.
- If this is a pre-release check the box `set as pre-release`
- Click on `publish release`
## Questions & Help

If you have any questions or require any help, please add your question in the 
[Contributer Q&A of AutoRA Discussions](https://github.com/orgs/AutoResearch/discussions/categories/contributor-q-a).
We look forward to hearing from you!
