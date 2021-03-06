import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random 
import pandas as pd

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

	for i in range(0,len(noneCount)):
		print("i: ",i, 'eC: ', 'nC: ',noneCount[i])
	return knownSet,unknownSet

#kset is a list of planets known to be in the habitable exoplaents
kset,uset = cleanRead()


#boosted is a new list of planets generated by picking each feature value randomly from kset
kkset = list()
ptemp = list()
uuset = list()
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

kset = normSet(kkset)
uset = normSet(uuset)


boosted = boostSet(kset,1000)
boosted = normSet(boosted)

#now label this dataset for supervised learning
#if in kset then add 1 or boosted. if in uset add 0
for planet in kset:
	planet.append(1.0)
for planet in uset:
	planet.append(0.0)
for planet in boosted:
	planet.append(1.0)

allset = boosted + kset + uset

X,y = dataLabelSplit(allset)




