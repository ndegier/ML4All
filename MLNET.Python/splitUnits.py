import os.path

filePathAlles = '/Users/aaikoosters/Documents/Develop/TensorFlow_python/CodeProjects/lifecycleSet/train_FD001.txt'

basePath = os.getcwd() + '/'
filename = 'lifecycleSet/train_FD001.txt'
filenameUni1 = 'lifecycleSet/uni1_train.txt'
fileSavePath = basePath + 'units/'
print(fileSavePath)


def writeToFile(listOfUnits, unitNumber):
    txtFile = open(fileSavePath + 'train_unit{}.txt'.format(unitNumber), 'w')
    print(txtFile)
    for x in listOfUnits:
        txtFile.write(x)
    txtFile.close()
        
unitLines = []
with open(basePath+filename) as fp:
   line = fp.readline()
   unitNumber = 1
   while line:
       words = line.split()
       if int(words[0]) != unitNumber:
           writeToFile(unitLines, unitNumber)
           unitNumber += 1
           unitLines = []
        # end if
       lines = line.strip()
    #    print("A {}: {}".format(words[0], lines))
       line = fp.readline()
       unitLines.append(lines)