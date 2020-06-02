import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random 
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
#############################Basically Junk but I use the length of earth row for indexing through features#############
earth_pl_hostname = 'sun'
earth_pl_letter = 'e'
earth_pl_pnum = 8.0
earth_pl_orbper = 365.0
earth_pl_orbsmax = 1.00000011
earth_pl_orbeccen = .0167
earth_pl_orbincl = 0.0
earth_pl_bmassj = .0031463520
earth_pl_radj = .08921402444
earth_pl_dens = 5.52
earth_pl_ra = 0 #not found yet
earth_pl_dec = 0#not found
earth_st_dist = 0.000004848
earth_st_optmag = 0#not found
earth_st_teff = 5777
earth_st_mass = 1
earth_st_rad = 1

#Earths values I found 
earthrow = [earth_pl_hostname,earth_pl_letter,earth_pl_pnum,earth_pl_orbper,earth_pl_orbsmax,earth_pl_orbeccen,
			earth_pl_orbincl,earth_pl_bmassj,earth_pl_radj,earth_pl_dens,
			earth_pl_ra,earth_pl_dec,earth_st_dist,earth_st_optmag,earth_st_teff,
			earth_st_mass, earth_st_rad]
#######################################################
def cleanRead():
	'''
	Reads planets.csv only pulling the features defined in featuers[]
	Returns two lists: knownSet-All planets defined in habitablePlanets[] 
					   unknownSet-Any planet not in habitablePlanets[]
	'''
	file = 'planets.csv'
	features = ['pl_hostname','pl_letter','pl_pnum', 'pl_orbper','pl_orbsmax','pl_orbeccen','pl_orbincl',
					'pl_bmassj','pl_radj','pl_dens','ra',
					'dec','st_dist','st_optmag','st_teff','st_mass','st_rad']
	noneCount = [0] *len(features)
	unknownSet = list()#planets unknown
	knownSet = list()#planets known to be in the habitiable zone
	planet = list()
	index = 0
	ii = 0
	with open(file) as f:
		reader = csv.DictReader(f)
		for row in reader:
			for i in range(0,len(features)):
				name = features[i]
				feat = row[name]
				if(feat == ''):
					noneCount[index]+= 1
					planet.append(0)
				else:
					planet.append(feat)
				index+= 1 #used to count number of useless features
				ii += 1
			
			if(ishabitable(planet)):
				knownSet.append(planet)
			else:
				unknownSet.append(planet)
			planet = list()
			index = 0
	return knownSet,unknownSet




def dataLabelSplit(planets):
	#return X, y where X is data y is labels
	X = list()
	y = list()
	for planet in planets:
		X.append(planet[:len(planet) - 1])

	for planet in planets:
		y.append(planet[-1])
	return X, y

'''
Found list a planets on Wikipedia that are in the "habitable zone" 
Used in cleanRead()
'''
#tuples of pl_hostname(star) and pl_letter(planet) => (star,planet)
habitablePlanets = [('Kepler-442','b'),('Wolf 1061','c'),('Kepler-452','b'),('Kepler-1229','b'),('Kepler-62','f'),('Kepler-186','f'),('Trappist-1','d'),('Kepler-1638','b'),
('Kepler-62','f'),('Kepler-186','f'),('Kepler-438','b'),('Kepler-296','e'),('Kepler-62','e'),('K2-3','d'),('Kepler-1554','b'),('Kepler-283','c'),('Kepler-440','b'),('HD 40307','g'),('K2-18','b'),('Kepler-61','b'),('Kepler-443','b'),('Kepler-22','b'),('Kepler-296','f'),
('Kepler-174','d'),('HD 20794','e'),('HD 219134','g'),('Kepler-1090','b'),('Kepler-298','d')]


def ishabitable(planet):
	for i in range(len(habitablePlanets)):
		if(planet[0] == habitablePlanets[i][0]):
			if(planet[1] == habitablePlanets[i][1]):
				return 1
	return 0


def boostSet(planetSet,epoch):
	'''
	Creates epoch number of new planets, each feature is taken at random from PlanetSet
	'''
	boosted = list()
	for i in range(epoch):
		planet = list()
		for j in range(len(planetSet[0])):
			index = random.randrange(0,len(planetSet) - 1)#chose planet at random
			planet.append(planetSet[index][j]) #append feature j, from planetSet[index]
		boosted.append(planet)

	return boosted


def normSet(planets):
	#for each feature col in set, collect the maxium value of the col and append to maxlist.
	#then for each planet feautre divide by the feature max
	#i loop planets
	maxx = -99999
	maxlist = list()
	ptemp = list()
	normSet = list()
	for feat in range(0,len(planets[0])):
		for planet in range(0,len(planets)):
			if(planets[planet][feat] > maxx):
				maxx = planets[planet][feat]
		maxlist.append(maxx)

	for i in range(0,len(planets)):
		for j in range(0,len(planets[i])):
			ptemp.append((float)(planets[i][j])/maxlist[j])
		normSet.append(ptemp)
		ptemp = list()
	return normSet


def shuffleData(planets,epoch = 10):
	for i in range(0,epoch):
		random.shuffle(planets)
	return planets

def addLabel(planets,label):
	for planet in planets:
		planet.append(label)
	return planets



def getHitOrMissAccuracy(pred,y_test):
	cor = 0.0
	incor = 0.0
	for i in range(0,len(pred)):
		 if pred[i] == y_test[i]:
		 	cor += 1
		 else:
		 	incor += 1
	return cor / (incor+cor)


#MAIN######

		#kset is a list of planets known to be in the habitable exoplaents

print("Reading Data from planet.csv")
kset,uset = cleanRead()



kkset = list()
ptemp = list()
uuset = list()

print("Converting all feature data to floats")
for planet in kset:
	for i in range(2,len(planet)):
		ptemp.append(float(planet[i]))
	kkset.append(ptemp)
	ptemp = list()
for planet in uset:
	for i in range(2,len(planet)):
		ptemp.append(float(planet[i]))
	uuset.append(ptemp)
	ptemp = list()

print("Normalizing Data")
kset = normSet(kkset)
uset = normSet(uuset)

			#boosted is a new list of planets generated by picking each feature value randomly from kset
print("boosting the dataset with 1000 positive examples")
boosted = boostSet(kset,1000)
boosted = normSet(boosted)

	#now label this dataset for supervised learning
	#if in kset then add 1 or boosted. if in uset add 0

print("Applying binary labels to the known planets habitable planets and all other planets as 0.0")
kset = addLabel(kset,1.0)
uset = addLabel(uset,0.0)
boosted = addLabel(boosted,1.0)


#Merge all the sets together to eventualy produce the end goal of X,y for scikit learn
allset = boosted + kset + uset

print("Shuffling data 50 times")
#shuffle data so learners have constant exposure to all types of data

allset = shuffleData(allset,50) #calls random.shuffle 50 times

#Break set of planets vectors into a list of planets and a list of labels. 
X,y = dataLabelSplit(allset)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.33,random_state=42)

#choosing a random forest Regrssor because I belive a prediction of how likely to be "habitiable" is more flexible and useful than a binary HABITABLE or NOT HABITABLE
forest = RandomForestClassifier(bootstrap=True,max_depth=3,max_features='auto',n_estimators=100)
mlp = Perceptron(random_state =1, alpha =1e-5,max_iter=450,tol=1e-3)
mlp.fit(X_train,y_train)
forest.fit(X_train,y_train)

X_test = np.asarray(X_test)

y_test = np.asarray(y_test)
forestPredictions = forest.predict(X_test)
mlpPredictions = mlp.predict(X_test)
print("Perceptron accuracy ",getHitOrMissAccuracy(mlpPredictions,y_test))
print("Forest accuaracy ",getHitOrMissAccuracy(forestPredictions,y_test))
