import numpy as np
import cv2

cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

person01 = np.load('Adriel.npy').reshape(100,50*50*3)

person02 = np.load('JC.npy').reshape(20,50*50*3)

#person03 = np.load('Ross Geller.npy').reshape(20,50*50*3)
#person04 = np.load('Rachel Greene.npy').reshape(20,50*50*3)
#person05 = np.load('Monica Geller.npy').reshape(20,50*50*3)
#person06 = np.load('Chandler Bing.npy').reshape(20,50*50*3)
#person07 = np.load('Joey Tribbiani.npy').reshape(20,50*50*3)
#person08 = np.load('Phoebe Buffay.npy').reshape(20,50*50*3)




names = {
        0:'AJ',
        1:'JC',
        #3:'Ross Geller',
        #4:'Rachel Greene',
        #5:'Monica Geller',
        #6:'Chandler Bing',
        #7:'Joey Tribbiani',
        #8:'Phoebe Buffay'
        }

data = np.concatenate([person01,person02])
labels = np.zeros((120,1))
labels[:100,:] = 0.0
labels[100:,:] = 1.0   
        
def distance(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())

def knn(testinput,data,labels,k):
    numRows = data.shape[0]
    dist = []
    for i in range(numRows):
        dist.append(distance(testinput,data[i]))
    dist = np.asarray(dist)
    index = np.argsort(dist)
    sortedLabels = labels[index][:k]
    counts = np.unique(sortedLabels,return_counts=True)
    return counts[0][np.argmax(counts[-1])]

"""sampleTest = [7,1,8,2,9,3]
sampleTest = np.asarray(sampleTest).reshape(3,2)

sampleLabel = np.array([1,1,0])
sampleInput = np.array([3,4]).reshape(1,2)

knn(sampleInput,sampleTest,)"""
    
while True:
    ret,frame = cap.read()
    grayFace = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayFace,1.3,5)
    
    for(x,y,w,h) in faces:
            croppedFace = frame[y:y+h,x:x+w]
            resizedFace = cv2.resize(croppedFace,(50,50))
            prediction = knn(resizedFace.flatten(),data,labels,5)
            name = names[int(prediction)]
            cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,255),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('Face Recognition',frame)
    if cv2.waitKey(1) == 27:
            break
cap.release()
cv2.destroyAllWindows()