<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/6295/6295417.png" width="100" />
</p>
<p align="center">
    <h1 align="center"></h1>
</p>
<p align="center">
    <em><code>► Fake Turkish New Classification with LSTM</code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/akatakan/Fake-News-Detection-Turkish?style=flat&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/akatakan/Fake-News-Detection-Turkish?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/akatakan/Fake-News-Detection-Turkish?style=flat&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/akatakan/Fake-News-Detection-Turkish?style=flat&color=0080ff" alt="repo-language-count">
<p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/HTML5-E34F26.svg?style=flat&logo=HTML5&logoColor=white" alt="HTML5">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
</p>
<hr>

##  Quick Links

> - [ Overview](#-overview)
> - [ Features](#-features)
> - [ Repository Structure](#-repository-structure)
> - [ Modules](#-modules)
> - [ Getting Started](#-getting-started)
>   - [ Installation](#-installation)
>   - [ Running ](#-running-)
> - [ License](#-license)

---

##  Repository Structure

```sh
└── /
    ├── clean2.csv
    ├── news_dataset.csv
    └── src
        ├── __pycache__
        │   └── app.cpython-39.pyc
        ├── app
        │   ├── LSTM.py
        │   ├── clean.csv
        │   ├── fake.py
        │   ├── preprocessing.py
        │   ├── real.py
        │   └── tr.py
        ├── app.py
        ├── best_model.h5
        ├── haber_tespit_model.h5
        ├── model
        │   ├── __init__.py
        │   ├── __pycache__
        │   │   ├── __init__.cpython-39.pyc
        │   │   ├── prediction.cpython-39.pyc
        │   │   ├── prediction_model.cpython-39.pyc
        │   │   ├── preprocess.cpython-39.pyc
        │   │   └── preprocessing.cpython-39.pyc
        │   ├── prediction_model.py
        │   └── preprocess.py
        ├── static
        │   └── styles
        │       └── style.css
        └── templates
            └── index.html
```

---

##  Modules

<details closed><summary>src</summary>

| File                                                                                     | 
| [app.py](https://github.com/akatakan/Fake-News-Detection-Turkish/blob/master/src/app.py) |

</details>

<details closed><summary>src.templates</summary>

| File                                                                                                       |
| ---                                                                                                        | 
| [index.html](https://github.com/akatakan/Fake-News-Detection-Turkish/blob/master/src/templates/index.html) | 

</details>

<details closed><summary>src.model</summary>

| File                                                                                                                     |
| [prediction_model.py](https://github.com/akatakan/Fake-News-Detection-Turkish/blob/master/src/model/prediction_model.py) | 
| [preprocess.py](https://github.com/akatakan/Fake-News-Detection-Turkish/blob/master/src/model/preprocess.py)             | 

</details>

<details closed><summary>src.app</summary>

| File                                                                                                             |
| ---                                                                                                              |
| [LSTM.py](https://github.com/akatakan/Fake-News-Detection-Turkish/blob/master/src/app/LSTM.py)                   | 
| [preprocessing.py](https://github.com/akatakan/Fake-News-Detection-Turkish/blob/master/src/app/preprocessing.py) |
| [real.py](https://github.com/akatakan/Fake-News-Detection-Turkish/blob/master/src/app/real.py)                   |
| [tr.py](https://github.com/akatakan/Fake-News-Detection-Turkish/blob/master/src/app/tr.py)                       |
| [fake.py](https://github.com/akatakan/Fake-News-Detection-Turkish/blob/master/src/app/fake.py)                   |

</details>

---

##  Getting Started

***Requirements***

Ensure you have the following dependencies installed on your system:

* **Python**: `version 3.10.6`

###  Installation

1. Clone the  repository:

```sh
git clone https://github.com/akatakan/Fake-News-Detection-Turkish/
```

2. Change to the project directory:

```sh
cd 
```

3. Install the dependencies:

```sh
pip install -r requirements.txt
```

###  Running 

Use the following command to run :

```sh
flask --app app run
```

---

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Submit Pull Requests](https://github.com/akatakan/Fake-News-Detection-Turkish/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/akatakan/Fake-News-Detection-Turkish/discussions)**: Share your insights, provide feedback, or ask questions.
- **[Report Issues](https://github.com/akatakan/Fake-News-Detection-Turkish/issues)**: Submit bugs found or log feature requests for .

<details closed>
    <summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine using a Git client.
   ```sh
   git clone https://github.com/akatakan/Fake-News-Detection-Turkish/
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.

Once your PR is reviewed and approved, it will be merged into the main branch.

</details>

---
