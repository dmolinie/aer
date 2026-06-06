Installing the AER package
++++++++++++++++++++++++++

.. Contents::


Prerequisites
=============

The AER package requires the following software:

1) **Python** 3.12.x or newer.

2) **PortAudio** To play audio sounds in the GUI, the ``portaudio`` library must be installed on the system. On Ubuntu:

  .. code-block:: sh

     sudo apt install libportaudio2

**Python** https://www.python.org/

**PortAudio** https://www.portaudio.com/


Installation
============

Except the optional packages specified below, the ``aer`` project relies on standard Python libraries only. The project's modules that depend on packages that are not installed on the system will not appear as tab-completion options (but may still be accessed via direct call).


Basic Installation
------------------

To install the AER package, when in the installation folder, run the following command:

.. code-block:: sh

   pip install .

This will compile the package's sources and install them into the active environment (e.g. "site-packages").


Additional Packages
-------------------

The options are:

* ``torch`` to install the Torch module:

  .. code-block:: sh

     pip install .[torch]

  By default, it is the torch default device type (*cpu* or *cuda*) that is installed; to specify between ``torch-cpu`` and ``torch-cuda``, use the requirement files provided in the package:

  .. code-block:: sh

     # To install torch-cpu  
     pip install .[torch] -r torch_cpu.txt

     # To install torch-cuda  
     pip install .[torch] -r torch_cuda.txt

* ``gui`` to install the ``sounddevice`` package, required to play audio sounds when using the GUI (``torch`` is still required to use the Torch models):

  .. code-block:: sh

     pip install .[gui]

* ``aidge`` to install the ``requests`` package; note that the ``aidge`` modules are not installed automatically and should be installed manually before using the AER's ``aidge`` submodule:

  .. code-block:: sh

     pip install .[aidge]

This will download and install the additional packages in addition to the ``aer`` package.
