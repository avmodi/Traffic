class TrafficDataReader():
	
	def __init__(self, fileName):
		self.file=fileName

	def read(self):
		file=self.file
		fp=open(file,'r')
		data=[]
		seqlen = list()
		for line in fp:
			lineArray=line.rstrip().split('\t')
			timeStampList=lineArray[1].replace('[','').replace(']','').split(',')
			durationList=lineArray[2].replace('[','').replace(']','').split(',')
			#print(timeStampList)
			#print(durationList)
			localData=[]
			for j in range(len(timeStampList)):
				eventFeed=[]
				eventFeed.append(int(durationList[j]))
				#eventFeed.append(0)
				eventFeed.append(int(timeStampList[j]))
				localData.append(eventFeed)
			data.append(localData)
			seqlen.append(len(localData))
		return data

def ignore_cong_durn(data):
	data_new = list()
	for loc in data:
		loc_new = [i[1] for i in loc]
		data_new.append(loc_new)

	return data_new


def get_interval(data):
	data_interval = list()
	i=0
	for loc in data:
		loc_new = list()
		loc_new.append([loc[0][0],0]) # First event timestamp is zero
		prev_ts = loc[0][1]
		for e in loc[1:]:
			loc_new.append([e[0],e[1]-prev_ts])
			prev_ts = e[1]
		data_interval.append(loc_new)
		if i<3: # Print old and new sequence for first three locations
			print(loc)
			print(loc_new)
			i+=1

	return data_interval


#readerObject=TrafficDataReader('/home/avmodi/RnD/pune-congestions.csv')
#print(readerObject.read())
