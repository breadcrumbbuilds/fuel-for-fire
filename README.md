# FUEL FOR FIRE

Welcome to fuel for fire, an experimental research project from TRU. Our vision is to predict wildfires, but you've got to walk before you can run. Follow these steps to get setup!

## Igniting the Spark (Setup)

Assuming you already have a decent grasp of Python, a favourite editor to work in, and a healthy amount of passion, we can get started.

• First things first, you're going to need [Anaconda](https://www.anaconda.com/distribution/).

_Note: If you're strapped for storage and prefer something lighter weight, go with [Miniconda](https://docs.conda.io/en/latest/miniconda.html)._

• Download for the appropriate operating system and follow the installation instructions.

• Once your download and installation of conda is complete, test to see that the installation was successful.

• In your terminal, type `conda env list`

• You should see an output for your base environment. If conda is not a recognized command, you'll need to add conda to the PATH

If you aren't familiar with environments, it's recommended that you do some reading: [Soft Introduction to Environments](https://medium.com/@monipip3/virtual-environments-explained-by-a-python-beginner-693a79b195da), [Conda Docs on Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

### Folder Structure

For some of these scripts to work, you'll need to adhere to a strict folder structure, or risk the scripts failing. Structure follows:

Working Directory  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-dev  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-(scripts)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-envs  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-(yaml files for environment setup)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-images  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-raw  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-<raw image files (.bin, .hdr)>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-converted  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-<images converted from binary (.png, .jpg)>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-colormap  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-<images for colormap (.png, .jpg)>

### Mac OSX

_TODO_

### Windows

_TODO_

## Creating Environments

_TODO_
