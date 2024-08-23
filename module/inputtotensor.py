import numpy as np
import torch

def inputtotensor(inputtensor, labeltensor):
    inputtensor = np.array(inputtensor)
    inputtensor = torch.FloatTensor(inputtensor)

    labeltensor = np.array(labeltensor)
    labeltensor = labeltensor.astype(float)
    labeltensor = torch.LongTensor(labeltensor)

    return inputtensor, labeltensor