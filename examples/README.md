# Example IMAGINE scripts

This directory contains two example IMAGINE scripts 
which complement the tutorials.

* `basic_pipeline.py`
   - Contains the same setup as the ["Basic elements..."][tutorial_one] tuorial,
     but runs using both `MultiNestPipeline` and `UltraNestPipeline`. 
     Both these pipelines support MPI, which allows to speed up the computation.
* `example_pipeline.py`
   - Contains a setup similar to ["Example pipeline"][tutorial_wmap], 
     but also including a stochastic field component and varying one parameter
     associated with it. The script is meant to be run in to step:
     executing it first to prepare (and test) the pipeline, and 
     using it to later run the pipeline.
* `example_pipeline.batch`
   - Contains an example of [SLURM][slurm_wiki] batch script which could be 
     used to start IMAGINE runs of `example_pipeline.py` in a
     typical HPC environment. 


[tutorial_wmap]: https://github.com/IMAGINE-Consortium/imagine/blob/master/tutorials/tutorial_wmap.ipynb
[tutorial_one]: https://github.com/IMAGINE-Consortium/imagine/blob/master/tutorials/tutorial_one.ipynb
[slurm_wiki]: https://en.wikipedia.org/wiki/Slurm_Workload_Manager