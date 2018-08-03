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






