# About this template

Hi, I've adapted this template from the excellent [python-project-template](https://github.com/rochacbruno/python-project-template/) by [rochacbruno](https://github.com/rochacbruno). It was created having in mind UKP Lab people and what the most common use-cases would be. Following its structure you'll get into developing your next paper in no time!

It includes:

- 📦 A basic [setup.py](setup.py) file to provide installation, packaging and distribution for your project; 
- 📃 Documentation structure using [mkdocs](http://www.mkdocs.org);
- 🧪 Testing structure using [pytest](https://docs.pytest.org/en/latest/);
- ✅ Code linting using [pylint](https://pypi.org/project/pylint/);
- 🎯 Entry points to execute your program using `python -m <murano>` with basic CLI argument parsing;
- 🔄 Continuous integration using [Github Actions](https://github.com/UKPLab/murano/actions) with jobs to check, lint and test your project;
- 🌐 An out-of-the-box project page created automatically for your project.

Are there any changes you'd like to request? Feel free to fork and open a pull request!

## Structure

Lets take a look at the structure of this template:

```text
│   .gitignore                      # A list of files to ignore when pushing to GH
│   ABOUT_THIS_TEMPLATE.md          # The file you're reading right now
│   LICENSE                         # The license for the project
│   mkdocs.yml                      # Configuration for documentation site
│   NOTICE.txt                      # Legal notice for the repository
│   README.md                       # The main readme for the project
│   requirements-dev.txt            # List of requirements for testing and devlopment
│   requirements.txt                # An empty file to hold the requirements for the project
│   setup.py                        # The setup.py file for installing and packaging the project
│   index.html                      # The template page for the automatically generated project page 
│
├───.github                         # Github metadata for repository
│   │   dependabot.yml              # Dependabot workflow for updating requirements
│   │   init.sh                     # Initializes the repository
│   │   PULL_REQUEST_TEMPLATE.md    # Used automatically by GH for pull requests
│   │   rename_project.sh           # Called once at repository creation
│   │
│   ├───ISSUE_TEMPLATE              # Templates for creating issues on GH 
│   │
│   └───workflows                   # GH Actions folder
│           docs.yml                # Builds documentation automatically
│           main.yml                # Runs install and file checks
│           rename_project.yml      # Renames repository at creation
│           tests.yml               # Run all tests in 'tests' folder
│
├───docs                            # Auto-generated documentation 
│       index.md                    # Landing page of docs
│
├───static                          # Images & CSS files to generate the project page 
│
├───murano             # The main python package for the project
│       base.py                     # The base module for the project
│       cli.py                      # Defines CLI instructions
│       __init__.py                 # This tells Python that this is a package
│       __main__.py                 # The entry point for the project
│
└───tests                           # Unit tests for the project (add more tests files here)
        conftest.py                 # Configuration, hooks and fixtures for pytest
        test_base.py                # The base test case for the project
        __init__.py                 # This tells Python that this is a test package
```

## FAQs

### How can I use this template to deploy a website for my paper ?

First, you should change the content of `index.html` to fit your needs. Some tinkering with the CSS inside `static` may be required. Then, you need to go to Settings > Pages > Build and deployment and select for Source "Deploy from a branch" and for Branch the one containing your `index.html` (by default it's the `main` branch).

### How can I adapt an existing project to deploy a project page ?

Add the `static` folder and `index.html` from this repository to your paper. Then, follow the FAQ above.

### I already have an external project page. Can I link my repo to it ?

Good news, you can display a repository website right next to your description. Simply click the ⚙ button next to the repo description and fill out the Website field.

### Where should I add new stuff ?

You should create new files and subpackages inside murano and implement your functionalities there. Remember to add what you write to `__init__.py` so that the imports work smoothly. Take a look at `base.py` and `__init__.py` to understand how it works.

### Why is `requirements.txt` empty ?

This template is a low dependency project, so it doesn't have any extra dependencies.
You can freely add new dependencies.

You should put here everything needed to replicate your work. 
Testing, linting, and other requirements used only in development should go in `requirements-dev.txt`.

### Why is there a `requirements-dev.txt` file ?

This file lists all the requirements for testing and development. Use it to separate things you used during development from the essential stuff needed to replicate your work.

### What is the `.github` folder?

It contains [GitHub Actions](https://docs.github.com/en/actions) that are executed automatically when pushing your code. You can see results for your repository [here](https://github.com/UKPLab/murano/actions).

### What does the linter workflow do?

It checks whether your code is clean enough from duplication, inconsistencies, violations to the naming convention etc.
It's not supposed to fail, but you should still look into it to get an idea of which parts of your code may need adjustments.

### Why do automated actions fail ?

This means there is something wrong in the files/tests/requirements. 
Click on the failing run to read more details.

### Why include `tests` and `docs` as part of the release?

This template ships with everything you may need. You can remove what you don't like in this way:
    - If you don't need automatic documentation generation, you can delete folder `docs`, file `.github\workflows\docs.yml` and `mkdocs.yml`
    - If you don't want automatic testing, you can delete folder `tests` and file `.github\workflows\tests.yml`

### How can I use pytest & pylint to check my code?

Command `pytest` called from the project folder will run all tests inside the `tests` folder.
Similarly, `pylint` will run linting checks on your code and give you a status report.
It checks things such as logic, formatting, correct imports, duplication etc. 

### Why conftest includes a go_to_tmpdir fixture?

When your project deals with file system operations, it is a good idea to use
a fixture to create a temporary directory and then remove it after the test.

Before executing each test pytest will create a temporary directory and will
change the working directory to that path and run the test.

So the test can create temporary artifacts isolated from other tests.

After the execution Pytest will remove the temporary directory.

### Why this template is not using [pre-commit](https://pre-commit.com/) ?

pre-commit is an excellent tool to automate checks and formatting on your code.

However I figured out that pre-commit adds extra dependency and it an entry barrier
for new contributors.

Once the project is bigger and complex, having pre-commit as a dependency can be a good idea.
