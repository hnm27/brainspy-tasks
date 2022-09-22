# brains-py tasks

This package shows a set of examples on how the libraries of the brainspy framework (brainspy and brainspy-smg) can be used.

Visit https://github.com/BraiNEdarwin/brains-py/wiki for more information.

![Insert image](https://raw.githubusercontent.com/BraiNEdarwin/brains-py/master/docs/figures/packages.png)

## 1. General description

The code offered in this package can perform the following tasks:

- Boolean classifier:
  - A single boolean gate classifier
  - VC-Dimension tests
  - Capacity tests for checking the performance on multiple VC-Dimensions
- Ring classifier:
  - A particular ring classifier with a given separation gap
  - Multiple runs on a ring classifier with a given separation gap
  - Capacity tests for checking the performance on multiple separation gaps for several runs

## 2. Jupyter notebooks

This repository includes a set of jupyter notebooks in brainspy-tasks/notebooks. Make sure you have installed brainspy and brainspy-smg before running them.

### 2.1 Installation

You can prepare the code in conda as follows:

`conda create -n bspy python==3.9`

`conda activate bspy`

`pip install brainspy brainspy-smg`

`conda install jupyter`

`cd brainspy-tasks/notebooks`

After the installation, remember to get out the environment and back again:

`conda deactivate`

`conda activate bspy`

Then, you can run:

`jupyter-notebook`

### 2.2 Examples available

- Checking device functionality: Basic steps on how to characterise a DNPU device and how to find adequate IV curves.
  
- Finding functionality on hardware: An example on how to use the genetic algorithm for quickly benchmarking the performance of a single DNPU device directly in hardware.
  
- Finding functionality on software: A simple example on how to train a custom model with a single device, for resolving the ring classification task.
  
- Advanced examlple on software: A simple example on how to train a custom model to resolve MNIST, using an architecture that is based on LeNet.
  

## 3. License and libraries

This code is released under the GNU GENERAL PUBLIC LICENSE Version 3. Click [here](https://github.com/BraiNEdarwin/brainspy-tasks/blob/master/doc/LICENSE) to see the full license.
The package relies on the following libraries:

- brainspy
- brainspy-smg
- jupyter
- tensorboard

## 4. Acknowledgements

This package has been created and it is maintained by the [Brains](https://www.utwente.nl/en/brains/) team of the [NanoElectronics](https://www.utwente.nl/en/eemcs/ne/) research group at the University of Twente. It has been designed and developed by:
This package has been created and it is maintained by the [Brains](https://www.utwente.nl/en/brains/) team of the [NanoElectronics](https://www.utwente.nl/en/eemcs/ne/) research group at the University of Twente. It has been designed by:

- **Dr. Unai Alegre-Ibarra**, [@ualegre](https://github.com/ualegre) ([u.alegre@utwente.nl](mailto:u.alegre@utwente.nl)): Project lead, including requirements, design, implementation, maintenance, linting tools, testing and documentation (Jupyter notebooks, Wiki and supervision of file by file documentation).
- **Dr. Hans Christian Ruiz-Euler**, [@hcruiz](https://github.com/hcruiz) ([h.ruiz@utwente.nl](mailto:h.ruiz@utwente.nl)): Initial design and implementation of major features both in this repository and in the legacy [SkyNEt](https://github.com/BraiNEdarwin/SkyNEt) repository and in this one.

With the contribution of:

- **Marcus Boon**: [@Mark-Boon](https://github.com/Mark-Boon): The on-chip gradient descent. The initial structure for the CDAQ to NiDAQ drivers in the legacy [SkyNEt](https://github.com/BraiNEdarwin/SkyNEt) repository.
- **Dr. ir. Michel P. de Jong** [@xX-Michel-Xx](https://github.com/xX-Michel-Xx) ([m.p.dejong@utwente.nl](mailto:m.p.dejong@utwente.nl)): Testing and identification of bugs, especially on the installation procedure.
- **Florentina Min Joo Uitzetter**: The genetic algorithm as shown in the legacy [SkyNEt](https://github.com/BraiNEdarwin/SkyNEt) repository.
- **Antonio J. Sousa de Almeida** [@ajsousal](https://github.com/ajsousal) ([a.j.sousadealmeida@utwente.nl](mailto:a.j.sousadealmeida@utwente.nl)): Checking and upgrading drivers and National Instruments equipment from the labs.
- **Bram van de Ven**, [@bbroo1](https://github.com/bbroo1) ([b.vandeven@utwente.nl](mailto:b.vandeven@utwente.nl)) : General improvements and testing of the different hardware drivers and devices and documentation.
- **Mohamadreza Zolfagharinejad** [@mamrez](https://github.com/mamrez) ([m.zolfagharinejad@utwente.nl](mailto:m.zolfagharinejad@utwente.nl)): Writing of some of the examples in Jupyter notebooks (IV curves and surrogate model generation).

Some of the code present in this project has been refactored from the [skynet](https://github.com/BraiNEdarwin/SkyNEt) legacy project. The original contributions to the scripts, which are the base of this project, can be found at skynet, and the authorship remains of those people who collaborated in it. Using existing scripts from skynet, a whole new structure has been designed and developed to be used as a general purpose python library.

This project has received financial support from:

- **University of Twente**
- **Dutch Research Council**
  - HTSM grant no. 16237
  - Natuurkunde Projectruimte grant no. 680-91-114
- **Horizon Europe research and innovation programme**
  - Grant no. 101046878
- **Toyota Motor Europe N.V.**
