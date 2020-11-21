import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from math import pi
from statistics import mean
from itertools import product

#Specimen Dimensions
L = 0.180##Input the bar length from the lab manual##  # Length of Bar [m]
R = 0.00238##Input the bar radius from the lab manual##  # Radius of the Bar [m]
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

data_list = {"Aluminum 1": [67.9, 0.2, 300.4, Aluminum1, 'Aluminum Sample #1', 374, 0.0417], "Aluminum 2": [67.9, 0.2, 300.4, Aluminum2, "Aluminum Sample #2", 374, 0.0417],
             "Steel 1": [150.5, 0.26, 334.4, Steel1, "Steel Sample #1", 779, 0.194], "Steel 2": [150.5, 0.26, 334.4, Steel2, "Steel Sample #2", 779, 0.194]}


class Rod():

    def __init__(self, E, nu, s_y, data, name, H, n):
        self.E = E
        self.nu = nu
        self.s_y = s_y
        self.data = data
        self.name = name
        self.H = H
        self.n = n
        self.G_th = (self.E * 1e9)/2/(1+self.nu) # in Pa
        self.tau_y = (self.s_y * 1e6)/(3**0.5)   # in Pa

        self.data['Angle (rad)'] = self.data['Angle (deg)'] * pi/180
        self.data['Theoretical Yield Radius (m)'] = (self.tau_y * L) / (self.G_th * self.data['Angle (rad)'])
        self.data['Force (N)'] = self.data['Force (kgf)'] * 9.81
        self.data['Torque (Nm)'] = self.data['Force (N)'] * (213 / 85) * 0.0523

        # Initialize some empty variables
        eTorque = []
        pTorque = []
        theoreticalTorque = []

        for i in range(len(self.data['Angle (rad)'])):
            # Create dummy variables to make the code look cleaner
            r_y = self.data['Theoretical Yield Radius (m)'].values[i]
            theta = self.data['Angle (rad)'].values[i]

            # Run the functions we defined above
            eTorque += [self.elasticTorque(r_y, theta)]  # The += here is adding values to the list, e.g. a=[1]; a+=[2] -> a=[1,2]
            pTorque += [self.plasticTorque(r_y, theta)]
            theoreticalTorque += [self.elasticPlasticTorque(r_y, theta)]

        # And include the values in a Pandas dataframe
        self.data['Theoretical Elastic Torque (Nm)'] = eTorque
        self.data['Theoretical Plastic Torque (Nm)'] = pTorque
        self.data['Theoretical Torque (Nm)'] = theoreticalTorque

    # Finish these functions for the elastic and plastic components of torque
    def elasticTorque(self, r_y, theta):
        if r_y > R:
            T_Elastic =  (self.G_th * theta * J) / L
        else:
            T_Elastic = (pi * self.tau_y * (r_y ** 3)) / 2
        return T_Elastic

    def plasticTorque(self, r_y, theta):
        if r_y > R:
            T_Plastic =  0
        else:
            T_Plastic =  ((2 * pi * self.H) / ((3 ** 0.5) * (self.n + 3))) * ((theta / ((3 ** 0.5) * L)) ** self.n) * ((R ** (self.n +3)) - (r_y ** (self.n + 3 )))
        return T_Plastic

    def elasticPlasticTorque(self, r_y, theta):
        T = self.elasticTorque(r_y, theta) + self.plasticTorque(r_y, theta)
        return T

    def maxShearStress(self, torque, R, J):
        tauMax = (torque * R) / J  ##equation for the max shear stress based on the torque in the elastic region##
        return tauMax

    def maxShearStrain(self, twist, R, L):
        gammaMax = (twist * R) / L   ##equation for the max shear strain based on the twist in the elastic region##
        return gammaMax

    def shearModulusFit(self, a, b):
        '''This is a linear fit to data between the data indices for a and b. Note, this will
        return an error if a or b are outside the length of Strain and Stress.'''

        # Find shear stress and shear strain
        shearStrain = [self.maxShearStrain(x, R, L) for x in self.data['Angle (rad)'][a:b]]
        shearStress = [self.maxShearStress(x, R, J) for x in self.data['Torque (Nm)'][a:b]]

        # Fit the modulus
        G, C, r, P, Err = linregress(shearStrain, shearStress)  # The data outputs the slope (G), intercept (C), regression (r) value, P-value and standard error
        # Note: Python lets you save multivariable outputs with a comma, i.e. a,b=[1,2] will give a=1 and b=2

        # Make a line for the fit data
        Y = [0.0, max(shearStress)]  # this is a list of length 2 for plotting the fit data later
        X = [(y - C) / G for y in
             Y]  # these are points that you can plot to visualize the data being fit, inverted from y=G*x+C, x=(y-C)/G
        return G, r, X, Y

    def shearModulusPlot(self, a, b):

        # Find the max shear stress and strain from elastic analysis
        shearStrain = [self.maxShearStrain(x, R, L) for x in self.data['Angle (rad)'][a:b]]
        shearStress = [self.maxShearStress(x, R, J) for x in self.data['Torque (Nm)'][a:b]]

        # Fit the modulus
        G, C, r, P, Err = linregress(shearStrain, shearStress)  # The data outputs the slope (G), intercept (C), regression (r) value, P-value and standard error
        # Note: Python lets you save multivariable outputs with a comma, i.e. a,b=[1,2] will give a=1 and b=2
        # Plot the max shear stress and strain and fit

        # Make a line for the fit data
        Y = [0.0, max(shearStress)]  # this is a list of length 2 for plotting the fit data later
        X = [(y - C) / G for y in
             Y]  # these are points that you can plot to visualize the data being fit, inverted from y=G*x+C, x=(y-C)/G

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

    def yield_rad_twist_plot(self):
        r_list = [R] * len(self.data['Angle (rad)'])
        # find index before r_y = r
        i = np.argwhere(np.diff(np.sign(self.data['Theoretical Yield Radius (m)'] - r_list))).flatten()
        theta_inter = round(float(self.data['Angle (rad)'][i]), 3)
        r_y_inter = round(float(self.data['Theoretical Yield Radius (m)'][i]) * 1000, 3)
        print(i)
        # Plot it to visualize
        plt.figure(figsize=(12, 8))
        plt.plot(self.data['Angle (rad)'], self.data['Theoretical Yield Radius (m)'], label='Yield Radius')
        plt.plot(self.data['Angle (rad)'], [R] * len(self.data['Angle (rad)']), label='Bar Radius')
        plt.plot(self.data['Angle (rad)'][i], self.data['Theoretical Yield Radius (m)'][i], 'ro',
                 label='Point before r_y > r' + '\n' + 'theta=' + str(theta_inter) + 'rad')
        plt.ylim(-0.0002, 0.003)
        plt.xlim(0, 3.14)
        plt.xlabel('Twist Angle(radian) ', fontsize=16)
        plt.ylabel('Radius', fontsize=16)
        plt.title('Yield Radius vs Twist Angle', fontsize=20)
        plt.legend()
        plt.show()


def interval(list):
    avg = mean(list)
    error = (max(list) - min(list)) / 2
    value = "{} +/- {}".format(avg, error)
    return value

def combined_torque_twist_plot(rod_list):
    plt.figure(figsize=(12, 8))
    plt.xlabel('Twist Angle (rad) ', fontsize=16)
    plt.ylabel('Torque (Nm)', fontsize=16)
    plt.title('Torque vs Twist Angle', fontsize=20)

    for rod in rod_list:
        plt.plot(rod.data['Angle (rad)'], rod.data['Torque (Nm)'], label=rod.name)

    plt.legend()
    plt.show()

def shear_mod_fit(rod_list, a, b):
    G_list = []
    r_list = []

    for rod in rod_list:
        G, r, X, Y = rod.shearModulusFit(a, b)
        G_list.append(G)
        r_list.append(r)

    G_list = [(x / 1e9) for x in G_list ]
    G_out = interval(G_list)
    r_out = interval(r_list)
    print("G: " + G_out)
    print("R: " + r_out)


rod_list = [Rod(*data_list[data]) for data in data_list]

rod_list[2].yield_rad_twist_plot()
'''
for rod in rod_list:
    print(rod.shearModulusPlot(0, 12))


combined_torque_twist_plot(rod_list[0:2])
combined_torque_twist_plot(rod_list[2:4])

shear_mod_fit(rod_list[0:2], 0, 12)
shear_mod_fit(rod_list[2:4], 0, 12)
'''