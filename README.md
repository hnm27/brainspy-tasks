
# brains-py tasks #
A python package based on the brains-py library to peform benchmark tests and tasks for studying the capacity of the boron-doped silicon devices. The package is part of the brains-py project, a set of python libraries to support the development of nano-scale in-materio hardware neural-network accelerators.

*   [![Tools](https://img.shields.io/badge/brainspy--black.svg)](https://github.com/BraiNEdarwin/brains-py): A python package to support the study of Dopant Network Processing Units as hardware accelerators for non-linear operations. Its aim is to support key functions for hardware setups and algorithms related to searching functionality on DNPUs and DNPU architectures both in simulations and in hardware.
 *   [![Tools](https://img.shields.io/badge/brainspy-smg-darkblue.svg)](https://github.com/BraiNEdarwin/brainspy-smg): A python package for creating surrogate models of nano-electronic materials.


![Insert image](https://raw.githubusercontent.com/BraiNEdarwin/brains-py/master/doc/figures/packages.png)


## 1. General description ##
This package supports the following tasks:
* Boolean classifier:
	* A single boolean gate classifier
	* VC-Dimension tests
	* Capacity tests for checking the performance on multiple VC-Dimensions
* Ring classifier:
	* A particular ring classifier with a given separation gap
	* Multiple runs on a ring classifier with a given separation gap
	* Capacity tests for checking the performance on multiple separation gaps for several runs

## 2. Installation instructions ##
The installation instructions differ depending on whether if you want to install as a developer or as a user of the library. Please follow the instructions that are most suitable for you:
* [User instructions](https://github.com/BraiNEdarwin/brains-py/blob/master/doc/USER_INSTRUCTIONS.md)
* [Developer instructions](https://github.com/BraiNEdarwin/brains-py/blob/master/doc/DEVELOPER_INSTRUCTIONS.md)

## 3. License and libraries ##
This code is released under the GNU GENERAL PUBLIC LICENSE Version 3. Click [here](https://github.com/BraiNEdarwin/brainspy-tasks/blob/master/doc/LICENSE) to see the full license.
The package relies on the following libraries:
* Pytorch
* Numpy
* Nidaqmx
* pyyaml
* Pyro4
* tqdm
* torch-optimizer
* pywin32

## 4. Acknowledgements
This package has been created and it is maintained by the [Brains](https://www.utwente.nl/en/brains/) team of the [NanoElectronics](https://www.utwente.nl/en/eemcs/ne/) research group at the University of Twente. It has been designed and developed by:
-   **Unai Alegre-Ibarra**, [@ualegre](https://github.com/ualegre) ([u.alegre@utwente.nl](mailto:u.alegre@utwente.nl))
-   **Hans Christian Ruiz-Euler**, [@hcruiz](https://github.com/hcruiz) ([h.ruiz@utwente.nl](mailto:h.ruiz@utwente.nl))

With the contribution of:
-  **Bram van de Ven**, [@bbroo1](https://github.com/bbroo1) ([b.vandeven@utwente.nl](mailto:b.vandeven@utwente.nl)) : General improvements and testing of the different hardware drivers and devices.
- **Michel P. de Jong** [@xX-Michel-Xx](https://github.com/xX-Michel-Xx) ([m.p.dejong@utwente.nl](mailto:m.p.dejong@utwente.nl)): Testing of the package and identification of bugs.
 - **Jochem Wildeboer** [@jtwild](https://github.com/jtwild/)  Perturbation ranking, Gradient ranking and Patch mapping tasks.

Some of the code present in this project has been refactored from the [skynet](https://github.com/BraiNEdarwin/SkyNEt) legacy project. The original contributions to the scripts, which are the base of this project, can be found at skynet, and the authorship remains of those people who collaborated in it. Using existing scripts from skynet, a whole new structure has been designed and developed to be used as a general purpose python library.  
