import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from math import pi

#Specimen Dimensions
L = 0.180##Input the bar length from the lab manual##  # Length of Bar [m]
R = 0.00476##Input the bar radius from the lab manual##  # Radius of the Bar [m]
J = (pi * (R ** 4)) / 2##Calculate the polar 2nd moment of area## #Polar 2nd moment of area [m^4]

#File with all the data
dataFile = 'Torsion data Aut2020.xlsx'

#Import the Steel Data
Steel1 = pd.read_excel(dataFile,header = 3,sheet_name = 'Data 1',usecols = [0,1])
Steel2 = pd.read_excel(dataFile,header = 3,sheet_name = 'Data 2',usecols = [0,1])

#Import the Aluminum Data
Aluminum1 = pd.read_excel(dataFile,header = 3,sheet_name = 'Data 1',usecols = [3,4])
Aluminum2 = pd.read_excel(dataFile,header = 3,sheet_name = 'Data 2',usecols = [3,4])

#Fix some weird stuff with the aluminum import
Aluminum1.columns = Steel1.columns #Rename the columns
Aluminum1 = Aluminum1.dropna() #Drop NaN values
Aluminum2.columns = Steel2.columns #Rename the columns
Aluminum2 = Aluminum2.dropna() #Drop NaN values

data_list = {"Aluminum 1": [67.9, 0.2, 300.4, Aluminum1, 'Aluminum Sample #1'], "Aluminum 2": [67.9, 0.2, 300.4, Aluminum2, "Aluminum Sample #2"],
             "Steel 1": [150.5, 0.26, 334.4, Steel1, "Steel Sample #1"], "Steel 2": [150.5, 0.26, 334.4, Steel2, "Steel Sample #2"]}


class Rod():

    def __init__(self, E, nu, s_y, data, name):
        self.E = E
        self.nu = nu
        self.s_y = s_y
        self.data = data
        self.name = name
        self.data['Angle (rad)'] = self.data['Angle (deg)'] * pi/180
        self.data['Force (N)'] = self.data['Force (kgf)'] * 9.81
        self.data['Torque (Nm)'] = self.data['Force (N)'] * (213 / 85) * 0.0523

    def maxShearStress(self, torque, R, J):
        tauMax = (torque * R) / J  ##equation for the max shear stress based on the torque in the elastic region##
        return tauMax

    def maxShearStrain(self, twist, R, L):
        gammaMax = (twist, R, L)   ##equation for the max shear strain based on the twist in the elastic region##
        return gammaMax

    def shearModulusFit(self, twist, torque, a, b):
        '''This is a linear fit to data between the data indices for a and b. Note, this will
        return an error if a or b are outside the length of Strain and Stress.'''

        # Find shear stress and shear strain
        shearStrain = [self.maxShearStrain(x, R, L) for x in twist[a:b]]
        shearStress = [self.maxShearStress(x, R, J) for x in torque[a:b]]

        # Fit the modulus
        G, C, r, P, Err = linregress(shearStrain,
                                     shearStress)  # The data outputs the slope (G), intercept (C), regression (r) value, P-value and standard error
        # Note: Python lets you save multivariable outputs with a comma, i.e. a,b=[1,2] will give a=1 and b=2

        # Make a line for the fit data
        Y = [0.0, max(shearStress)]  # this is a list of length 2 for plotting the fit data later
        X = [(y - C) / G for y in
             Y]  # these are points that you can plot to visualize the data being fit, inverted from y=G*x+C, x=(y-C)/G
        return G, C, r, X, Y

    def shearModulusPlot(self, a, b):
        G, C, r, X, Y = self.shearModulusFit(Steel1['Angle (rad)'].values, Steel1['Torque (Nm)'].values, R, L, J, a, b)

        # Find the max shear stress and strain from elastic analysis
        shearStrain = [self.maxShearStrain(x, R, L) for x in self.data['Angle (rad)'][a:b]]
        shearStress = [self.maxShearStress(x, R, J) for x in self.data['Torque (Nm)'][a:b]]

        # Plot the max shear stress and strain and fit
        plt.figure(figsize=(12, 8))
        plt.plot(shearStrain, shearStress, '.-')
        plt.plot(X, Y, '--', label='Modulus Fit G=' + str(round(G * 1e-9, 2)) + 'GPa')
        plt.xlabel('Max Shear Strain', fontsize=16)
        plt.ylabel('Max Shear Stress (Pa)', fontsize=16)
        plt.legend()
        plt.show()

    def torque_twist_plot(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.data['Angle (rad)'], self.data['Torque (Nm)'], '.-')
        plt.xlabel('Twist Angle (rad) ', fontsize=16)
        plt.ylabel('Torque (Nm)', fontsize=16)
        plt.title('Torque vs Twist Angle', fontsize=20)
        plt.legend()
        plt.show()

rod_list = [Rod(*data_list[data]) for data in data_list]


def torque_twist_plot(rod_list):
    plt.figure(figsize=(12, 8))
    plt.xlabel('Twist Angle (rad) ', fontsize=16)
    plt.ylabel('Torque (Nm)', fontsize=16)
    plt.title('Torque vs Twist Angle', fontsize=20)

    for rod in rod_list:
        plt.plot(rod.data['Angle (rad)'], rod.data['Torque (Nm)'], label=rod.name)

    plt.legend()
    plt.show()

torque_twist_plot(rod_list[0:2])
torque_twist_plot(rod_list[2:4])

