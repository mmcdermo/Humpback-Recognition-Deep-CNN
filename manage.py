
# We must import skimage.io before we import any other imaging libraries,
#  or PIL will be unable to load images for us.
#  This is a known bug in scikit image. 
import skimage.io

import sys
import inspect
import argparse
from experiment import Experiment
import experiments

def addExperimentParams(experiment, parser):
    """
    Add environmental parameters for an experiment to a command line parser.
    """
    for param in experiment.listEnvParams():
        req = True if "required" in param else False
        parser.add_argument("--"+param["name"], help=param["desc"], required=param["required"])
    return parser

def experimentClasses():
    """
    Get all experiment classes from experiments
    """
    experimentObjs = {}
    for name, obj in inspect.getmembers(experiments, inspect.isclass):
        experimentObjs[name] = obj
    return experimentObjs
    
def experimentFromCommandline(experimentName):
    """
    Create an experiment from commandline parameters
    """
    classes = experimentClasses()
    if experimentName not in classes:
        print("Please select a known experiment: ")
        for k in classes:
            print("\t-"+k)
    exp = classes[experimentName]()
    parser = argparse.ArgumentParser(description='Experiment parameters')
    addExperimentParams(exp, parser)
    args, unknown = parser.parse_known_args(sys.argv[1:])
    args = vars(args)
    envParams = {}
    for p in exp.listEnvParams():
        envParams[p["name"]] = args[p["name"]]
    return (exp, envParams)

def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(description='Experiment manager.')

    parser.add_argument("--experiment", help="Which experiment you'd like to run", required=True)
    parser.add_argument("--loadTrial", help="Which trial you'd like to load")
    
    modeGroup = parser.add_mutually_exclusive_group(required=True)
    modeGroup.add_argument("--train", help="Use with --experiment or --loadTrial to train the model", action='store_const', const=True, default=False)
    modeGroup.add_argument("--evaluate", help="Use with --loadTrial to evaluate performance", action='store_const', const=True, default=False)
    modeGroup.add_argument("--predict", help="Use with --loadTrial to predict the given image")

    if len(sys.argv) < 2:
        parser.print_help()
        print("Available Experiments: ")
        for k in experimentClasses():
            print("\t-"+k)
        exit(1)

    # Permit unknown arguments since we'll use them as experiment parameters
    args, unknown = parser.parse_known_args(sys.argv[1:])
    args = vars(args)

    if "experiment" in args:
        exp, envParams = experimentFromCommandline(args["experiment"])
        print(envParams)

        trialName = None
        if "loadTrial" in args and args["loadTrial"]:
            trialName = args["loadTrial"]
            
        if args["train"]:
            exp.run(envParams)
        if args["predict"]:
            exp.envParams = envParams
            predictions = exp.predictImage(args["predict"], trialName)
        elif args["evaluate"]:
            exp.envParams = envParams
            performance = exp.evaluate()
            pass
        return
    
if __name__ == "__main__":
    main()
