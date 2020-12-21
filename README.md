# DanQ: A hybrid convolutional and recurrent deep neural network for quanitfying the function of DNA seuqences (re-implemented)
An updated implementation of the DanQ neural network, originally published in 2015 by Quang, D. and Xie, X. The paper can be found here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4914104/ and their code base here: https://github.com/uci-cbcl/DanQ.

In this version, we re-implement DanQ using modern versions of the Keras framework.

# Install
DanQ was originally implemented in version 0.2.0 of Keras. This version is considered very old amongst machine learning toolkits and is no longer usable. As such, the original DanQ code base is nearly impossible to run as is, since details of dependences are not provided.

To run this re-implemented version, it is highly recommended you set up a virtual environment. At the time of this writing, the latest version available for Python is 3.9.x but TensorFlow (the backend used for Keras) only runs for Python 3.8.x. You will need Python 3.8.x to run this implementation. Using the correct version you can create a virtual environment with the following command on OSX/Linux:
```
[PATH_TO_PYTHON_3.8.x]/python -m venv danq-env
```

This will create a virtual environment with the right version of Python and Pip. Activate the environment:
```
source danq-env/bin/activate
```

Next, you will likely have to update Pip to ensure it finds all of the correct packages:
```
pip install --upgrade pip
```

Now, you can use the requirements.txt file to install all of the correct required packages:
```
pip install -r requirements.txt
```

Once installation completes, you should have the required libraries to run the application. However, it is highly recommended that you setup the necessary toolkits to make use of an Nvidia GPU if you have one. If you don't have one, you can skip this section, but if you do, you'll need the following:

[CUDA] (https://developer.nvidia.com/cuda-toolkit) Enables GPU use for training and evaluation. Make sure to install the correct version for your GPU drivers!

[cuDNN] (https://developer.nvidia.com/cuDNN) Significantly speeds up convolution operations. You will require an Nvidia developer's login, but this is free and can be done pretty quickly. Make sure to install the version that matches your GPU drivers AND the toolkit you used!

Once everything is setup, you can try to run the script in DEBUG mode which will run through a very small subset of the data to ensure everything is working. You can do this with:

```
python danq.py DEBUG
```

If everything works, you can now train the network on the full dataset by removing the 'DEBUG' argument. 






