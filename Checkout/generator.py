import numpy as np
import sys

class WeightSensors:
    '''
    Class that generates different values for the weight sensors
    '''
    def __init__(self, weightSensorPath, sep = '\n', number_sensors = 4):
        '''
        This method reads from a file a list of numbers that represents the sum of weight sensors values.
        '''
        self.__weightSensorPath = weightSensorPath
        self.__number_sensors = number_sensors

        with open(self.__weightSensorPath,'r') as weightFile:
            weightData = weightFile.read().split(sep)
            try:
                self.__weightValues = [float(value) for value in weightData]
                self.__current_weights = [self.__weightValues[0]/self.__number_sensors for _ in range(0, self.__number_sensors)]
                self.__current_index_weight = 0
                self.__history = []
            except ValueError:
                print('Not all values found in the file is a numerical one.')


    def generate(self, iterations = 3):
        '''
       This approach verifies if there are additional elements in the weight list. 
       If so, it divides them into sets of four values each, aiming to achieve a roughly equal sum for each set.
        '''
        assert self.__current_index_weight < len(self.__weightValues), "End of weights list reached."
        current_weights = [self.__weightValues[self.__current_index_weight]/self.__number_sensors for _ in range(0, self.__number_sensors)]
        self.__current_weights, history_weights = self.generate_weights_iterations(current_weights, iterations=iterations)
        self.__history += history_weights
        self.__current_index_weight += 1

    
    @property
    def current_weights(self):
        return self.__current_weights
    
    @property
    def history(self):
        return self.__history
    
    @property
    def number_sensors(self):
        return self.__number_sensors

    def generate_weights_iterations(self, weights, iterations = 3, changeThrs = 0.1):
        '''
        This method initiates by dividing the sensor sum into four parts and slightly adjusting each part by introducing a small amount of noise through a uniform distribution. 
        If the probability is below 1%, one or more elements can be expressed as any number within the range of negative infinity to positive infinity. 
        '''
        assert iterations >= 0, "The iteration count must be a non-negative value."
        new_weights = weights
        history_weights = [new_weights]
        for _ in range(iterations):
            change_amount = np.random.uniform(-abs(changeThrs), abs(changeThrs), self.__number_sensors)
            new_weights = new_weights + change_amount

            if np.random.rand() < 0.01:
                selected_sensors = np.random.choice(range(self.__number_sensors), size=np.random.randint(1, self.__number_sensors+1), replace=False)
                random_values = np.random.uniform(-sys.maxsize, sys.maxsize, len(selected_sensors))

                for index, value in zip(selected_sensors, random_values):
                    new_weights[index] = value
            
            history_weights.append(new_weights.tolist())
        
        return new_weights, history_weights

