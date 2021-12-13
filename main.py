import streamlit as st
import sys
import random
import numpy as np
import scipy.ndimage as nd
import tensorflow as tf
import pandas as pd

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show, saving
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform

sys.setrecursionlimit(10**6)

numNeurons = 832
imageRes = 128
choiceLogPath = "imageChoices.csv"

param_f = lambda: param.image(imageRes)

model = models.InceptionV1()
model.load_graphdef()


def getRandomObjective():
    """
    Returns a randomly generated objective
    """
    obj = 0
    weights = []
    for n in range(0,numNeurons):
        w = random.uniform(-1,1)
        obj += w * objectives.channel("mixed5a", n)
        weights.append(w)
    return obj

def getRandomBasisObjective():
    """
    Selects a random neuron to activate and returns and objective
    with only that single neuron activated.
    """
    neuronNum = int(random.uniform(0,numNeurons))
    return objectives.channel("mixed5a", neuronNum)

def getCircuitImage(objective):
    """
    Input an activation object and it will output a
    numpy array of the image.
    """
    res = render.render_vis(model, objective, param_f)
    print("res:",type(res[0]))
    return res[0]

def showImages():
    """ Display a random and basis image in a random order
    """
    images = [
        (getCircuitImage(getRandomBasisObjective()),"basis"),
        (getCircuitImage(getRandomObjective()),"rand")
    ]
    random.shuffle(images)
    st.write("Image 1")
    st.image(images[0][0])
    st.write("Image 2")
    st.image(images[1][0])
    return images

def writeToChoiceLog(choice):
    with open(choiceLogPath, "a") as choiceLog:
        choiceLog.write("\n")
        choiceLog.write(choice)

def chartChoiceLog():
    choices = pd.read_csv(choiceLogPath)

    st.bar_chart(choices["choice"].value_counts())

"""
# Circuit Interpretability Survey Test
"""

images = showImages()

with st.form("my_form"):
    choice = st.radio(
        "Which image is easier to interpret as a feature of an object, animal, or person?",
        ["Image 1", "Image 2"]
    )
    submitted = st.form_submit_button("Submit")

    # Log choice on submit
    if submitted:
        choiceIndex = 1
        if(choice == "Image 1"): choiceIndex = 0

        writeToChoiceLog(images[choiceIndex][1])

chartChoiceLog()