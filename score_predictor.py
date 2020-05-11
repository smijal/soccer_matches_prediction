#####################################################
############     |ECE 407 - PROJECT 1|    ###########
############     |  Stefan Mijalkov  |    ###########     
############     |    04/29/2020     |    ###########
#####################################################

#################################################################
### Decription: Predicts soccer winner based on team statistics #
### Dataset decription: Stats from soccer matches, including    #
###      yellow cards, shots on target, half-time goals ...     #
### Features used:                                              # 
### HTHG = Half time Home team goals                            #
### HTAG = Half time Away team goals                            #
### HS = Home team shots                                        #
### AS = Away team shots                                        #
### HST = Home team shots on target                             #
### AST = Away team shots on target                             #    
### HC = Home team corners                                      #    
### AC = Away team corners                                      #
### HR = Home team red cards                                    #
### AR = Away team red cards                                    #
### HTWinRate = Home team win rate                              #
### ATWinRate = Away team win rate                              #
### HTLoseRate = Home team lose rate                            #
### ATLoseRate = Away team lose rate                            #
### HTDrawRate = Home team draw rate                            #
### ATDrawRate = Away team draw rate                            #
### Target feature: FTR = Full Time Result                      #
### ML Models used: Logistic Regression & SVM                   #
#################################################################

#################################################################
### Extract the data in the same folder where the             ###  
### score_predictor.py is saved                               ###
#################################################################

import pandas as pd
import os 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import glob
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import svm

print("########################################################")
print("#######      Soccer winning team predictor       #######")
print("########################################################")  

####################### Datapath ##########################
#root = input("\nEnter datapath to folder: ") 
root = "C:/Users/hazar/Desktop/PL_score_prediction"
fileNames = os.listdir(root)
print("\nSelect league: ")

##############
seasons = ['2009-2010', '2010-2011','2011-2012','2012-2013','2013-2014',
          '2014-2015','2015-2016','2016-2017','2017-2018','2018-2019']

leagues = ['English-premier-league', 'Spanish-La-Liga', 'German-Bundesliga']

##############
for index, value in enumerate(leagues):
    print(str(index) + " -> " + value )

l = int(input("League: "))
league = leagues[l]


print("Select season to display stats about the home teams winning rate in the league: ")
for index, value in enumerate(seasons):
    print(str(index) + " -> " + value )

s = int(input("Season: "))
season = seasons[s].split('-')
season = str(season[0][2:4] + season[1][2:4])
print("")

l_folder = root + '/' + league
dataName = l_folder + '/' + "season-"+season+".csv"
data= pd.read_csv(dataName)


teams = list(set(data['HomeTeam']))
teams.sort()

### Just shows winning rate statistics for the home team
### Statistically the home team has better chances of winning
print("HOME TEAM HAS BETTER WINNING CHANCES")
print("\nTeams that played in " + str(league) + " in the season " + str(seasons[s]) + ": ")
print(teams)

num_matches = data.shape[0]
num_features = data.shape[1]

wins_home_team = np.count_nonzero(data['FTR'] == 'H')
loses_home_team = np.count_nonzero(data['FTR'] == 'A')
draws_home_team = np.count_nonzero(data['FTR'] == 'D')

print("\nHome team stats for the {} season:".format(season))
print ("Win rate {:.3f}%".format(float(wins_home_team / num_matches)*100))
print ("Lose rate {:.3f}%".format(float(loses_home_team / num_matches)*100))
print ("Draw rate {:.3f}%".format(float(draws_home_team / num_matches)*100))

plt.pie([wins_home_team, loses_home_team, draws_home_team], labels=['wins' , 'loses', 'draws'])
plt.show()


# # READS ALL THE DATA FOR THE SPECIFIED LEAGUE FOR PAST 10 SEASONS
allData=glob.glob(l_folder + "/*.csv")

bigData = pd.read_csv(allData[0])
for i in range(1,len(allData)):
    dat = pd.read_csv(allData[i])
    bigData=pd.concat([bigData, dat], ignore_index=True, sort=False)



storedData = bigData.copy(deep=True)
num_matches = bigData.shape[0]
wins_home_team = np.count_nonzero(bigData['FTR'] == 'H')
loses_home_team = np.count_nonzero(bigData['FTR'] == 'A')
draws_home_team = np.count_nonzero(bigData['FTR'] == 'D')

print("\nHome team stats in the past 10 seasons:")
print ("Win rate {:.3f}%".format(float(wins_home_team / num_matches)*100))
print ("Lose rate {:.3f}%".format(float(loses_home_team / num_matches)*100))
print ("Draw rate {:.3f}%".format(float(draws_home_team / num_matches)*100))

plt.pie([wins_home_team, loses_home_team, draws_home_team], labels=['wins' , 'loses', 'draws'])
plt.show()

cols = bigData.columns
teamDict={}
all_teams = []
for team in bigData['HomeTeam']:
    teamDict[team] = [0,0,0]
    all_teams.append(team)

#calculates number of wins, loses, and draws for a team
# I am calculating win/lose/draw rates for each team to improve my accuracy
# since the dataset is missing many valuable features 
for index, row in bigData.iterrows():
    if(row['FTR']=='H'):
        teamDict[row['HomeTeam']][0] += 1
        teamDict[row['AwayTeam']][1] += 1
    elif(row['FTR']=='A'):
        teamDict[row['HomeTeam']][1] +=1
        teamDict[row['AwayTeam']][0] +=1
    else:
        teamDict[row['HomeTeam']][2] +=1
        teamDict[row['AwayTeam']][2] +=1

# labels is the target value that we are trying to predict: H,A or D
labels = bigData['FTR']

#dropping unecessary features from the dataframe
count=0
for c in bigData.columns:
    if( not (c.startswith('H')) and  (not c.startswith('A')) ):
        bigData.drop(c, axis=1, inplace=True)


teamRates = {}
for team in all_teams:
    teamRates[team] = [0.0,0.0,0.0]

#displays stats about team name and win rate in the past 10 seasons
print("\nWin rates of different teams in the " + str(league) + str(":"))
for team in teamDict.keys():
    winRate = teamDict[team][0]/(380)*100
    loseRate = teamDict[team][1]/(380)*100
    drawRate = teamDict[team][2]/(380)*100
    teamRates[team][0] = winRate
    teamRates[team][1] = loseRate
    teamRates[team][2] = drawRate
    print(str(team) + ": {:.2f}%".format(winRate))

h_list=[]
a_list=[]
hd_list=[]
ad_list=[]
hl_list=[]
al_list=[]
for index, row in bigData.iterrows():
    h_list.append(teamRates[row['HomeTeam']][0])
    a_list.append(teamRates[row['AwayTeam']][0])
    hd_list.append(teamRates[row['HomeTeam']][2])
    ad_list.append(teamRates[row['AwayTeam']][2])
    hl_list.append(teamRates[row['HomeTeam']][1])
    al_list.append(teamRates[row['AwayTeam']][1])

#attaching calculated win/lose/draw rates in the dataframe
bigData['HTWinRate'] = h_list
bigData['ATWinRate'] = a_list
bigData['HTLoseRate'] = hl_list
bigData['ATLoseRate'] = al_list
bigData['HTDrawRate'] = hd_list
bigData['ATDrawRate'] = ad_list

#dropping unnecessary data,  
bigData.drop('HomeTeam', axis=1,inplace=True)
bigData.drop('AwayTeam', axis=1, inplace=True)
bigData.drop('HTR', axis=1, inplace=True)
bigData.drop('HF', axis=1, inplace=True)
bigData.drop('AF', axis=1, inplace=True)
bigData.drop('HY', axis=1, inplace=True)
bigData.drop('AY', axis=1, inplace=True)

print("\n################################")
print("Features used: ")
print(bigData.columns)

for col in bigData.columns:
    bigData[col] = scale(bigData[col])

# split the dataset into training set 90% and 10% test set
xTrain, xTest, yTrain, yTest = train_test_split(bigData.dropna(), labels.dropna(), test_size = 0.1, random_state = 45)

############################### Training models and prediction ####################################
print("\n##########################################")
print("STATS FROM LOGISTIC REGRESSION CLASSIFIER")
model = LogisticRegression(random_state=42,max_iter=1000)
model.fit(xTrain, yTrain)
actual = yTest.to_numpy()
predicted1 = model.predict(xTest)
score = model.score(xTest, yTest)*100
print("Accuracy: {:.2f}%".format(score))
cm1=confusion_matrix(yTest, predicted1, labels=['H', 'A', 'D'])
print("\nConfustion matrix:")
print(cm1)

print("\n##########################################")
print("STATS FROM SUPPORT VECTOR MACHINES")
model2 = svm.SVC()
model2.fit(xTrain, yTrain)
predicted2 = model2.predict(xTest)
score = model2.score(xTest, yTest)*100
print("Accuracy: {:.2f}%".format(score))
cm2=confusion_matrix(yTest, predicted2, labels=['H', 'A', 'D'])
print("\nConfustion matrix:")
print(cm2)

### Prints actual team names to display a better intuition of the match and
### teams that are playing.
j=0
yTest=yTest.to_numpy()
yTest = yTest.reshape(1,yTest.shape[0])
predicted1 = predicted1.reshape(1,predicted1.shape[0])
for i in xTest.index:
    print("\nMatch: ", end='')
    print(str(storedData['HomeTeam'][i]) + " vs. " + str(storedData['AwayTeam'][i]) )
    print("Predicted winner: ", end='')
    if(predicted1[0][j]=='H'):
        winner = storedData['HomeTeam'][i]
    elif(predicted1[0][j]=='A'):
        winner = storedData['AwayTeam'][i]
    else:
        winner="None/Draw"
    print(winner)

    print("Actual winner: " , end='')
    if(yTest[0][j]=='H'):
        winner = storedData['HomeTeam'][i]
    elif(yTest[0][j]=='A'):
        winner = storedData['AwayTeam'][i]
    else:
        winner="None/Draw"
    print(winner)

    j+=1
    k = input("Press enter to predict another match or E to exit...").lower() 
    if(k=='e'):
        exit()  