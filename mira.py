# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation
import util
PRINT = True

class MiraClassifier:
  """
  Mira classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mira"
    self.automaticTuning = False
    self.C = 0.001
    self.legalLabels = legalLabels
    self.max_iterations = max_iterations
    self.initializeWeightsToZero()

  def initializeWeightsToZero(self):
    "Resets the weights of each label to zero vectors"
    self.weights = {}
    for label in self.legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
  
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    "Outside shell to call your method. Do not modify this method."  
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
    
    if (self.automaticTuning):
        Cgrid = [0.002, 0.004, 0.008]
    else:
        Cgrid = [self.C]
        
    return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
    """
    This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
    then store the weights that give the best accuracy on the validationData.

    Use the provided self.weights[label] data structure so that
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    representing a vector of values.
    """
    scores = util.Counter()
    cweight = util.Counter()
    count = util.Counter()
    for c in Cgrid:
      hits = 0
      for iteration in range(self.max_iterations):

        print "Starting iteration " + str(iteration+1) + "..."

        for i in range(len(trainingData)):
          for j in self.legalLabels:
            scores[j] = trainingData[i] * self.weights[j]

          guess = scores.argMax()

          if not (trainingLabels[i] == guess):
            a = trainingLabels[i]
            b = trainingData[i]
            t = ( ( (self.weights[guess] - self.weights[a])*b + 1.0 )/(2.0*(b*b)) )
            maxt = min(c, t)
            varvector = util.Counter()
            for k in trainingData[i].keys():
              varvector[k] = trainingData[i][k] * maxt

            self.weights[trainingLabels[i]] += varvector
            self.weights[scores.argMax()] -= varvector

      cweight[c] = self.weights

      # Now run this against the validation data to see which c is the best
      for i in range(len(validationData)):
        for j in validationLabels:
          scores[j] = validationData[i] * self.weights[j]

        if validationLabels[i] == scores.argMax():
          hits += 1  # Add a count to each hit

      count[c] = hits  # Add the amount of hits to the respective c

    print "Using weight number " + str(count.argMax())
    self.weights = cweight[count.argMax()]  # Use the highest hit rate one

  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  def findHighOddsFeatures(self, label1, label2):
    """
    Returns a list of the 100 features with the greatest difference in feature values
                     w_label1 - w_label2

    """
    featuresOdds = []

    "*** YOUR CODE HERE ***"

    return featuresOdds

