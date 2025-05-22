import numpy as np

class ResilienceMetrics:
    """
    This class is designed to compute the resilience concept through a temporal ensemble
    of indicators. The construction of the object requires specifying the number of
    indicators or dimensions through which resilience will be assessed. Additionally,
    the construction of the class requires the indicator ensemble function.

    To execute this class, invoke the "fit" method, providing a list of arrays
    representing the system's performance signal (Pset), the reference signal (Rset)
    in the absence of disruptive events, and a list containing indices indicating the
    occurrence of disruptive events. This method returns the resilience metric.
    """
    def __init__(self, K, numberScenarios, assemblyIndicatorFuction):
        # Construction of class

        self.K = K # Number of indicators or dimensionalities of resiliencie
        self.numberScenarios = numberScenarios # Number of scenarios
        self.assemblyIndicatorFuction = assemblyIndicatorFuction # Funtion of assembly indicators
        self.Pset = {} # List of system's performance signals
        self.Rset = {} # List of reference signal (Rset) in the absence of disruptive events
        self.disturbancesIndex = {} # Dictionary index in scenary
                                    # of list of indices indicating the occurrence of disruptive events

        # Dictionaries for time windows of analysis. Index in (k, l, scenary), k indicator, l index of time windows
        self.timeWindowsDic = {} # Dictionary for system's performance signals
        self.referenceWindowsDic = {} # Dictionary for reference signal

        # Dictionaries for resilience summary metric in each time windows.
        # Index in (k, l, scenary), k indicator, l index of time windows and scenary the scenary of the measure
        self.summaryMetricsDic = {}

        # Milestones of resilience curve in the windows time.
        # Index in (k,l, scenary), k indicator, l index of time windows, scenary the scenary
        # The milestones is a tuple with (Ti, Tf, Tr)
        # Ti = incidence time
        # Tf = failure time
        # Tr = recovery time
        self.milestones = {}

        # Dictionary for the results of assembly time of resilience metrics
        self.assemblyTimeDic = {}

        # List of resilience in scenary
        self.scenaryResilience = [0 for i in range(numberScenarios)]

    def fit(self, disturbancesIndex, Pset, Rset):
      """ Method for measuring resilience across the stages of the methodology

          Parameters:
          disturbancesIndex: dictionary index by scenary indices indicating the occurrence of disruptive events in list or array []
          Pset: the system's performance signal in dictionary index by scenary and list of list or array [[], [], ..., []]
          Rset: the reference signal dictionary index by scenary with list of list or array [[], [], ..., []]

          Return:
          Resilience value assambly time and indicator
      """

      ####################################################
      # Stage I. Performance index and reference signals #
      ####################################################
      self.Pset = Pset
      self.Rset = Rset
      self.disturbancesIndex = disturbancesIndex

      #################################################################
      # Stage II. Find the time windows and calculated summary metric #
      #################################################################
      self.timeWindows() #Find the time windows
      self.summaryMetrics() #Calculated summary metric

      for scenary in range(self.numberScenarios):

        ###################################
        # Stage III. Time-window assembly #
        ###################################

        for k in range(self.K):
          vectorTimeSummaryMetrics = [self.summaryMetricsDic[(k, l, scenary)] for l in range(len(self.disturbancesIndex[scenary]))]
          if len(vectorTimeSummaryMetrics) > 1:
            self.assemblyTimeDic[(k, scenary)] = self.assemblyTime(vectorTimeSummaryMetrics)
          else:
            self.assemblyTimeDic[(k, scenary)] = vectorTimeSummaryMetrics[0]

        ###################################
        # Stage IV. Indicator assembly    #
        ###################################
        self.scenaryResilience[scenary] = self.assemblyIndicator(scenary)

      ##################################
      # Stage V. Scenary assembly      #
      ##################################

      return np.mean(self.scenaryResilience), np.std(self.scenaryResilience)

    ###################################
    # Stages and auxiliar methods     #
    ###################################

    def timeWindows(self):
        """
        In this method the time windows are identified by considering the
        disruptive event moment and adding time to the left and right to cover a window.
        """
        for scenary in range(self.numberScenarios):
          lenDisturbance = len(self.disturbancesIndex[scenary])
          if lenDisturbance == 0: # Not disruptive event
            assert False, 'There are not disruptive events.'

          elif lenDisturbance == 1: # One only disruptive event
              for k in range(self.K):
                self.timeWindowsDic[(k, 0, scenary)] = self.Pset[scenary][k]
                self.referenceWindowsDic[(k, 0, scenary)] = self.Rset[scenary][k]
                Ti =  self.disturbancesIndex[scenary][0]
                Tr =  len(self.Pset[scenary][0]) - 1
                Tf =  np.argmin(self.timeWindowsDic[(k, 0, scenary)][Ti+1:Tr])+ Ti + 1
                self.milestones[(k, 0, scenary)] = (Ti, Tf, Tr)

          else: # More than one disruptive event
            for l, i in enumerate(self.disturbancesIndex[scenary]):
              if l == 0: # First disruptive event
                for k in range(self.K):
                  self.timeWindowsDic[(k, l, scenary)] = self.Pset[scenary][k][0: self.disturbancesIndex[scenary][1] + 1]
                  self.referenceWindowsDic[(k, l, scenary)] = self.Rset[scenary][k][0: self.disturbancesIndex[scenary][1] + 1]
                  Ti =  i
                  Tr =  self.disturbancesIndex[scenary][l+1]
                  Tf =  np.argmin(self.timeWindowsDic[(k, l, scenary)][Ti+1:Tr])+ Ti + 1
                  self.milestones[(k, l, scenary)] = (Ti, Tf, Tr)

              elif l == len(self.disturbancesIndex[scenary]) - 1: # Last disruptive event
                for k in range(self.K):
                  self.timeWindowsDic[(k, l, scenary)] = self.Pset[scenary][k][self.disturbancesIndex[scenary][l]-1: len(self.Pset[scenary][k])]
                  self.referenceWindowsDic[(k, l, scenary)] =  self.Rset[scenary][k][self.disturbancesIndex[scenary][l]-1: len(self.Rset[scenary][k])]
                  Ti =  i  - self.disturbancesIndex[scenary][l] + 1
                  Tr = len(self.timeWindowsDic[(k, l, scenary)]) - 1
                  Tf = np.argmin(self.timeWindowsDic[(k, l, scenary)][Ti+1:Tr])+ Ti + 1
                  self.milestones[(k, l, scenary)] = (Ti, Tf, Tr)

              else: # Intermediate disruptive events
                for k in range(self.K):
                  self.timeWindowsDic[(k, l, scenary)] = self.Pset[scenary][k][self.disturbancesIndex[scenary][l]-1:self.disturbancesIndex[scenary][l+1]+1]
                  self.referenceWindowsDic[(k, l, scenary)] = self.Rset[scenary][k][self.disturbancesIndex[scenary][l]-1:self.disturbancesIndex[scenary][l+1]+1]
                  Ti =  i  - self.disturbancesIndex[scenary][l] + 1
                  Tr = self.disturbancesIndex[scenary][l+1] - self.disturbancesIndex[scenary][l] + 1
                  Tf = np.argmin(self.timeWindowsDic[(k, l, scenary)][Ti+1:Tr]) + Ti + 1
                  self.milestones[(k, l, scenary)] = (Ti, Tf, Tr)


    def summaryMetrics(self):
      """ For each time window, the corresponding resilience summary metric is computed
          for each indicator. The metric may be guided by equations found in the
          literature.
      """
      for key in self.timeWindowsDic:

        # Milestones in resilience curve
        Ti = self.milestones[key][0]
        Tf = self.milestones[key][1]
        Tr = self.milestones[key][2]
        deltaTf = Tf - Ti
        deltaTr = Tr - Tf

        # Failure profile
        FP = np.trapz((self.timeWindowsDic[key])[Ti:Tf+1], range(Ti, Tf+1))/ np.trapz((self.referenceWindowsDic[key])[Ti:Tf+1], range(Ti, Tf+1))

        # Recovery profile
        RP = np.trapz((self.timeWindowsDic[key])[Tf:Tr+1], range(Tf, Tr+1))/ np.trapz((self.referenceWindowsDic[key])[Tf:Tr+1], range(Tf, Tr+1)) # integral tiempo de flla hasta recuperación

        # Compute de summary metrics
        if np.isnan(FP) or np.isnan(RP):
          self.summaryMetricsDic[key] = 1e-10
        else:
          self.summaryMetricsDic[key] = (Ti + FP*deltaTf + RP*deltaTr) / (Ti + deltaTf + deltaTr)

    def assemblyTime(self, vectorTimeSummaryMetrics):
      """ In this method is computed the time-window assembly.
          An averaging across consecutive time windows is computed. This average
          penalized if a decrease is observed beetween periods and an incremental factor
          when it increases from period to period.

          Parameters:

          vectorTimeSummaryMetrics = list of summary metrics in each time window
      """
      print(vectorTimeSummaryMetrics)
      if len(vectorTimeSummaryMetrics) == 2: # Only two time windows
          resilienceTime = 0.5*(vectorTimeSummaryMetrics[0] + vectorTimeSummaryMetrics[1])*(1 + (vectorTimeSummaryMetrics[1] - vectorTimeSummaryMetrics[0]))
          if resilienceTime >= 1: resilienceTime = 1
          if resilienceTime < 0: resilienceTime = 1e-10
          return resilienceTime
      else: # More than two time windows
          meanMonthVariaton = [0.5*(vectorTimeSummaryMetrics[i] + vectorTimeSummaryMetrics[i+1])*(1 + (vectorTimeSummaryMetrics[i+1] - vectorTimeSummaryMetrics[i])) for i in range(len(vectorTimeSummaryMetrics)-1)]
          return self.assemblyTime(meanMonthVariaton)

    def assemblyIndicator(self, scenary):
      """ This method computed the coupling summary metrics by indicators. The way of
          coupling is set in the assemblyIndicatorFuction property of the object.
      """
      if self.assemblyIndicatorFuction == 'harmonic':
        inverseSum = 0
        for k in range(self.K):
          inverseSum += 1/self.assemblyTimeDic[(k, scenary)]
        return self.K/inverseSum

      if self.assemblyIndicatorFuction == 'mean':
        sum = 0
        for k in range(self.K):
          sum += self.assemblyTimeDic[(k, scenary)]
        return sum/self.K

      if self.assemblyIndicatorFuction == 'geometric':
        product = 1
        for k in range(self.K):
          product *= self.assemblyTimeDic[(k, scenary)]
        return (product)**(1/self.K)

      assert False, f'The assembly indicator function is not correctly defined for the object: {self}'
      