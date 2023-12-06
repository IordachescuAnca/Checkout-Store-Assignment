
class WeightDetector:
    '''
    Class designed to determine whether there has been any alteration in weight or if the weight has reached a stabilized state.
    '''
    def __init__(self, weightThrs, stabIntervalInputPath, sep='\n'):
        '''
        Method that reads from a file all the stabilization intervals.
        '''
        self.__weightThrs = weightThrs

        self.__stabIntervalInputPath = stabIntervalInputPath
        with open(self.__stabIntervalInputPath,'r') as stabInputFile:
            stabIntervals = stabInputFile.read().split(sep)
        
        self.__stabIntervals = [(float(interval.split(' ')[0]), float(interval.split(' ')[1])) for interval in stabIntervals]
        self.__indexInterval = 0


        self.__minInterval = self.__stabIntervals[self.__indexInterval][0]
        self.__maxInterval = self.__stabIntervals[self.__indexInterval][1]
        self.stability_tries = 0

    
    def check_weight_change(self, currentWeights, previousWeights):
        '''
        Method that assesses whether the disparity between two weight sensors is noteworthy.
        '''
        sum_current_weights = sum(currentWeights)
        sum_previous_weights = sum(previousWeights)
        if abs(sum_current_weights - sum_previous_weights) > self.__weightThrs:
            return True, currentWeights
        else:
            return False, None
        

    def check_stability(self, new_total_weight):
        '''
        Method that checks if the weight is stable.
        '''
        if self.__minInterval <= new_total_weight <= self.__maxInterval:
            self.stability_tries += 1
            return True
        else:
            self.stability_tries = 0
            return False
        
    
    def is_stable(self, num_tries):
        return self.stability_tries == num_tries
    
    
    def reset(self):
        '''
        A method that resets the detector in response to the reception of a weight by the server, signaling stability.
        '''
        self.stability_tries = 0
        self.__indexInterval += 1
        self.__indexInterval = min(self.__indexInterval, len(self.__stabIntervals)-1)
        self.__minInterval = self.__stabIntervals[self.__indexInterval][0]
        self.__maxInterval = self.__stabIntervals[self.__indexInterval][1]



