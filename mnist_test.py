import importlib.resources as importlib_resources
import jpype.imports
import traceback

jar_path = importlib_resources.files('pytetrad').joinpath('resources', 'tetrad-current.jar')
jar_path = str(jar_path)
jvm_args = ["-Xmx30G", "-Xms30G"]
    
if not jpype.isJVMStarted():
    try:
        #jpype.startJVM(jpype.getDefaultJVMPath(), "-Xmx2048m", convertStrings=False, classpath=[jar_path])
        #jpype.startJVM(jpype.getDefaultJVMPath(), classpath=[jar_path])
        jpype.startJVM(jpype.getDefaultJVMPath(), *jvm_args, f"-Djava.class.path={jar_path}")
    except OSError:
        print("can't load jvm")
        traceback.print_exc()
        pass

import pytetrad.tools.TetradSearch as ts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv
import cl4py


from cl4py import Symbol
from tensorflow.keras.datasets import mnist

def make_mnist_csv():
    print("Loading MNIST data")
    # Load the MNIST dataset
    (X_train, y_train), (_, _) = mnist.load_data()

    dfs = []
    for im_arr, label in zip(X_train, y_train):
        _, binary = cv.threshold(im_arr, 10, 255, cv.THRESH_BINARY)
        row = np.append(binary.flatten(),label)
        dfs.append(pd.DataFrame([row]))
    df = pd.concat(dfs, axis=0,ignore_index=True)
    df.rename(columns={784: "label"}, inplace=True)
    #df.to_csv("mnist_data.csv", index=False)
    return df


def init_hems():
    # get a handle to the lisp subprocess with quicklisp loaded.
    lisp = cl4py.Lisp(cmd=('sbcl', '--dynamic-space-size', '30000',
                           '--script'), quicklisp=True, backtrace=True)
    
    # Start quicklisp and import HEMS package
    lisp.find_package('QL').quickload('HEMS')
    
    # load hems and retain reference.
    hems = lisp.find_package("HEMS")
    return hems

def run_fci(df):
    # Get the Runtime instance
    runtime = jpype.JClass("java.lang.Runtime").getRuntime()

    # Get the total memory allocated to the JVM
    total_memory = runtime.totalMemory() / (1024 * 1024 * 1024)
    print(f"Total Memory allocated in Gigabytes: {total_memory}")

    ## Make a TetradSearch instance to run searches against. This helps to organize
    ## the use of Tetrad search algorithms and hides the JPype code for those who
    ## don't want to deal with it.
    search = ts.TetradSearch(df)
    search.set_verbose(False)

    ## Pick the score to use, in this case a continuous linear, Gaussian score.
    search.use_bdeu(sample_prior=10, structure_prior=0)
    search.use_chi_square(alpha=0.1)

    print('FCI')
    search.run_fci()
    print(search.get_string())


def run_gfci(df):
    # Get the Runtime instance
    runtime = jpype.JClass("java.lang.Runtime").getRuntime()

    # Get the total memory allocated to the JVM
    total_memory = runtime.totalMemory() / (1024 * 1024 * 1024)
    print(f"Total Memory allocated in Gigabytes: {total_memory}")

    ## Make a TetradSearch instance to run searches against. This helps to organize
    ## the use of Tetrad search algorithms and hides the JPype code for those who
    ## don't want to deal with it.
    search = ts.TetradSearch(df)
    search.set_verbose(False)

    ## Pick the score to use, in this case a continuous linear, Gaussian score.
    search.use_bdeu(sample_prior=10, structure_prior=0)
    search.use_chi_square(alpha=0.1)

    print('GFCI')
    #search.run_gfci(max_degree=4, max_disc_path_length=4, depth=4)
    search.run_gfci(max_degree = 16, max_disc_path_length=16, depth=16)
    print(search.get_string())

def main():
    #df = make_mnist_csv()
    df = pd.read_csv("~/Code/mimic-iv-tools/mimic_df.csv")
    run_gfci(df)

    #hems = init_hems()
    #net = hems.fci(hems.make_df(Symbol("mnist"), "mnist_data.csv"))
    #bindings = hems.make_edge_bindings(net.cdr)
    #print(bindings)
    
if __name__ == "__main__":
    main()
