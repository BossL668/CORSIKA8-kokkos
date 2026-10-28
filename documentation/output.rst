Output
======

The format of CORSIKA 8 is designed to allow simple and robust managmenet of large libraries, as well as high reading performance. There is a dedicated python library to help processing data.

The basic structure of output is structured on the filesystem itself with a couple of subdirectories and files. Each run of CORSIKA~8 creates a library that can contain any number of showers.
The format is equally suited for single huge showers as well as for a very high number of very low-energy showers. Each module included in the run can
produce output inside this directory. The module output is separeted in individual user-named sub-directories, each containing files produced by the module. The file format is either yaml for basic configuration and summary data, or Apache parquet for any other (binary, compressed)
data. Parquet is optimal for columnar/tabular data as it is produced by CORSIKA 8.

One advantage of this format is that with normal filesystem utilties users can manage the libraries. On all systems there are tools available to
directly read/process yaml as well as parquet files. If you, for example, don't need the particle data for space reasons, this is very simple to remove from a library. Individual
output stream (modules) can be easily separated with no extra effort.

For example, the output of the "vertical_EAS" example program looks like this:

::

  vertical_EAS_outputs/
      config.yaml
      summary.yaml
      particles/
          config.yaml
          summary.yaml
          particles.parquet

The "vertical_EAS_outputs" and the "particles" are user-defined names and can be arranged/changed. But the type
of data is well defined, e.g. in "particles" the data from an ObservationPlane object is stored. This is relevant,
since it allows python to access this data in a controlled way.

The top level "config.yaml" contains top-level library information:

.. code-block:: YAML

  name: vertical_EAS_outputs
  creator: CORSIKA8
  version: 8.0.0-prealpha

and the "summary.yaml" is written in the very end (thus, the presence of the summary also indicates that a run is finished):

.. code-block:: YAML

  showers: 2
  start time: 06/02/2021 23:46:18 HST
  end time: 06/02/2021 23:46:42 HST
  runtime: 00:00:24.260

Each module has its own "config.yaml" and "summary.yaml" file, too.
To handle thus output for analysis any tool of your preference is feasible. We recommend python. There is a python library accompanied with
CORSIKA~8 to facilitate analysis and output handling (>>> is python prompt):

.. code-block:: python

   >>> import corsika
   >>> lib = corsika.Library("vertical_EAS_outputs")
   >>> lib.config  # this gets the library configuration as a Python dictionary
   {'name': 'vertical_EAS_outputs',
   'creator': 'CORSIKA8',
   'version': '8.0.0-prealpha'}
   >>> lib.names  # get a list of all registered processes in the library
   ['particles']
   >>> lib.summary  # you can also load the summary information
   {'showers': 1,
   'start time': '06/02/2021 23:46:18 HST',
   'end time': '06/02/2021 23:46:30 HST',
   'runtime': 11.13}
   >>> lib.get("particles")  # you can then get the process by its registered name.
   ObservationPlane('particles')
   >>> lib.get("particles").config  # and you can also get its config as well
   {'type': 'ObservationPlane',
   'plane': {'center': [0, 0, 6371000],
   'center.units': 'm',
   'normal': [0, 0, 1]},
   'x-axis': [1, 0, 0],
   'y-axis': [0, 1, 0],
   'delete_on_hit': True,
   'name': 'particles'}
   >>> lib.get("particles").data  # this returns the data as a Pandas data frame
      shower  pdg        energy         x         y    radius
   0         0  211  9.066702e+10  2.449931 -5.913341  7.093710
   1         0   22  2.403024e+11 -1.561504 -1.276160  2.024900
   2         0  211  1.306354e+11 -4.626045 -3.237780  6.009696
   3         0  211  1.773324e+11 -1.566567  4.172961  4.461556
   4         0  211  7.835374e+10  3.152863 -1.049201  3.330416
   ..      ...  ...           ...       ...       ...       ...
   >>> lib.get("particles").astype("arrow")  # you can also request the data in a different format
   pyarrow.Table
   shower: int32 not null
   pdg: int32 not null
   energy: double not null
   x: double not null
   y: double not null
   radius: double not null
   >>>lib.get("particles").astype("pandas") # or astype("arrow"), or astype("pandas").to_numpy()

You can locally install the corsika python analysis library from within your corsika
source code directory by `pip3 install --user -e python pyarrow==0.17.0`. Note, the pyarrow
version fix has shown to be needed on some older systems. You may not need this, or you may
need additional packages, too.
