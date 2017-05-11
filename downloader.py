import csv
import urllib.request
import os
import socket

socket.setdefaulttimeout(60)
print(1)
filename = input("Filename...[without .csv]\n")
pos = int(input("From ?th...[0,1,2,3...]\n"))
file = open('dataset/'+filename+'.csv','r')
csvfile = csv.reader(file)


iterator = iter(csvfile)
counter = pos
for i in range(pos+1):
	next(iterator)
for row in iterator:
    print("Downloading file{}".format(counter))
    counter += 1
    try:
        if not os.path.isfile(os.path.join('dataset/{}/{}.jpg').format(filename,row[0])):
            urllib.request.urlretrieve(row[1],os.path.join('dataset/{}/{}.jpg').format(filename,row[0]))
    except Exception as e:
        print('Fail to download one object due to {}.Passed.'.format(e))
        logfile = open('log_{}.txt'.format(filename),'a+')
        logfile.write('{},{},{},{}\n'.format(counter,row[0],row[1],e))
        logfile.close()

file.close()
