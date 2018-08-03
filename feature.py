import time
from datetime import datetime

def getDatetime(epoch):
	date_string=time.strftime("%d-%m-%Y %H:%M:%S", time.localtime(epoch))
	date= datetime.strptime(date_string, "%d-%m-%Y %H:%M:%S")
	return date

def isWeekend(epoch):
	date=getDatetime(epoch)
        #print(date, date.weekday())
	if date.weekday() in (5,6):
		return 1
	else:
		return 0

def hourOfDay(epoch):
	date=getDatetime(epoch)
	oneHot=[0]*24
	oneHot[date.hour]=1
	return oneHot


def dayOfWeek(epoch):
	date=getDatetime(epoch)
	oneHot=[0]*7
	oneHot[date.weekday()]=1
	return oneHot

def monthOfYear(epoch):
	date=getDatetime(epoch)
	oneHot=[0]*12
	oneHot[date.month-1]=1
	return oneHot

def minutesOfHour(epoch):
	date=getDatetime(epoch)
	oneHot=[0]*60
	oneHot[date.minute]=1
	return oneHot

def dayOfMonth(epoch):
	date=getDatetime(epoch)
	oneHot=[0]*31
	oneHot[date.day-1]=1
	return oneHot

def gap(epoch1, epoch2, MinGap, MaxGap, isNormalize):
	diff=epoch2-epoch1
        #print(epoch1, epoch2, diff)
	if not isNormalize:
		if diff > 0:
			return diff/3600.0
		else :
			return diff*(-1.0)/3600.0
	else:
		return diff/(MaxGap-MinGap)

def moving_average(gaps):
	gaps_avg=0.0
	for i in range(len(gaps)):
		gaps_avg+=gaps[i]
	gaps_avg/=len(gaps)
	return gaps_avg

def exp_moving_average(gaps,alpha):
	gaps_avg=list()
	gaps_avg.append(gaps[0])
	for i in range(1,len(gaps)):
		gap_average=alpha*gaps[i] + (1.0-alpha)*gaps_avg[i-1]
		gaps_avg.append(gap_average)

	return gaps_avg[-1]
#print(dayOfWeek(1522635973))

def multiScaleAverage(gaps,alpha):
	# scales is a list of alphas
	weights=[0.0,0.1,0.3, 0.6]
	gaps_avg=list()
	for i in range(len(alpha)):
		gaps_avg.append(weights[i]*exp_moving_average(gaps,alpha=alpha[i]))

	return gaps_avg



