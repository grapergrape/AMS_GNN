import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

def calculate_curvatures(mesh_file):
    # Read the source file.
    reader = vtk.vtkOBJReader()
    reader.SetFileName(mesh_file) 
    reader.Update()
    polydata = reader.GetOutput()

    # Calculate Mean curvature
    curvaturesFilter = vtk.vtkCurvatures()
    curvaturesFilter.SetInputData(polydata)
    curvaturesFilter.SetCurvatureTypeToMean()
    curvaturesFilter.Update()
    mean_curvature = vtk_to_numpy(curvaturesFilter.GetOutput().GetPointData().GetArray("Mean_Curvature"))
    
    # Calculate Gaussian curvature
    curvaturesFilter.SetCurvatureTypeToGaussian()
    curvaturesFilter.Update()
    gaussian_curvature = vtk_to_numpy(curvaturesFilter.GetOutput().GetPointData().GetArray("Gauss_Curvature"))

    # Calculate principal curvatures
    sqrt_H2_K = np.sqrt(np.abs(mean_curvature**2 - gaussian_curvature))
    k1 = mean_curvature + sqrt_H2_K
    k2 = mean_curvature - sqrt_H2_K


    return mean_curvature, gaussian_curvature, k1, k2

