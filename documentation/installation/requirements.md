---
title: Requirements
layout: default
documentation: true
category_weight: 2
categories: [Installation]
---

{% include toc.html %}

## Supported OS

Currently, we provide support for

- macOS 10.12 "Sierra"
- Debian 9 "Stretch"
- Ubuntu 16.10 "Yakkety Yak"

which are known to enable the easy installation of Storm. Other Linux distributions are likely to work too, but it may take significant effort to get the required versions of the dependencies up and running. For example, thanks to [Joachim Klein](http://www.inf.tu-dresden.de/index.php?node_id=1473), there is a [script]({{ site.github.url }}/resources/scripts/installation/storm-build-debian-jessie.sh) that installs Storm and some crucial dependencies on Debian 8 "Jessie".

In the following, we will detail all dependencies of Storm and how to install them on the supported platforms.

## Dependencies

### Compiler

For the compilation step, a C++14-compliant compiler is required. Storm is known to work with

- GCC 5.3, GCC 6
- clang 3.5.0
- AppleClang 8.0.0

Newer versions of these compilers will probably work, but are not tested. In particular, the following list of compilers is known to *not* work.

- GCC versions 4.9.1 and older
- clang 3.4 and older

### General Dependencies

The following two lists provide an overview over the *required* and *recommended* dependencies of Storm. *Required* dependencies are absolutely essential for Storm to be compiled and must be installed. *Recommended* dependencies are optional, but not installing them may severely limit the offered functionality.

Required:
- git
- cmake
- boost (>= 1.61)
- cln
- ginac
- autoreconf
- doxygen
- glpk
- hwloc (on some systems)

Recommended:
- z3 (not strictly required, but already needed for standard tasks like PRISM/JANI model building)
- xercesc (installation prevents an expensive part of the build step)
- [MathSAT](http://mathsat.fbk.eu/) (needed by the abstraction refinement engine, needs to be configured manually during the [configuration](manual-configuration.html#mathsat)

## OS specific preparations

We collected some platform specific hints to ease the installation of Storm on the supported operating systems. Since Storm has some optional dependencies that enhance it's functionality, and some dependencies that are strictly required, we show how to install both the *required* and *recommended* dependencies. The installation instructions of the *recommended* dependencies are to be understood incrementally, i.e. in addition to the required dependencies.

### macOS 10.12 "Sierra"

First of all, you need to download and install Xcode and its command line utilities to have the suitable command line tools. For more details, we refer to a [this tutorial](https://www.moncefbelyamani.com/how-to-install-xcode-homebrew-git-rvm-ruby-on-mac/).

Furthermore, we recommend the usage of [Homebrew](https://brew.sh) to install the missing packages, but MacPorts might (at some point) have the desired dependencies as well.

- Required:
```
brew install cln ginac boost autoconf cmake doxygen
brew tap homebrew/science
brew install homebrew/science/glpk
```

- Recommended:
```
brew install z3 xerces-c
```

### Debian 9 "Stretch"

- Required:
```
sudo apt-get install git cmake libboost-all-dev libcln-dev libginac-dev automake doxygen libglpk-dev
```

- Recommended
```
sudo apt-get install libz3-dev libxerces-c-dev
```

### Ubuntu 16.10 "Yakkety Yak"

- Required:
```
sudo apt-get install git cmake libboost-all-dev libcln-dev libginac-dev automake doxygen libglpk-dev libhwloc-dev
```

- Recommended:
```
sudo apt-get install libz3-dev libxerces-c-dev
```
