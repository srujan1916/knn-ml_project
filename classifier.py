import numpy as np
from sklearn import model_selection
from sklearn.utils import resample
import csv
from math import sqrt,inf
#from imblearn.over_sampling import SMOTE   this didn't work


def get_data(ip_file): #reads input features and predicted values from csv
	inputs=[]       #features
	classes=[]  #
	i=0
	if ip_file=='cat4.csv':
		with open(ip_file) as csv_file:
			reader=csv.reader(csv_file)
			for row in reader:
				if i!=0:        #csv reader doesn't allow subscripting, so instead of [1:], this, to skip row with column names
					x=row[3:14]+row[15:-1]
					x=[float(i) for i in x]
					inputs.append(np.array(x))
					classes.append(row[14])
				i+=1
	elif ip_file=='cat1.csv':  #reading from cat1.csv
		with open(ip_file) as csv_file:
			reader=csv.reader(csv_file)
			for row in reader:
				if i!=0:        #csv reader doesn't allow subscripting, so instead of [1:], this, to skip row with column names
					x=row[3:14]+row[17:-7]
					x=[float(i) for i in x]
					inputs.append(np.array(x))
					classes.append(row[15])
				i+=1
      
	elif ip_file=='cat3.csv':
		with open(ip_file) as csv_file:
			reader=csv.reader(csv_file)
			for row in reader:
				if i!=0:        #csv reader doesn't allow subscripting, so instead of [1:], this, to skip row with column names
					x=row[3:14]+row[17:-7]
					x=[float(i) for i in x]
					inputs.append(np.array(x))
					classes.append(row[15])
				i+=1  
	else: #reading cat2
		with open(ip_file) as csv_file:
			reader=csv.reader(csv_file)
			for row in reader:
				if i!=0:        #csv reader doesn't allow subscripting, so instead of [1:], this, to skip row with column names
					x=row[4:15]+row[18:-7]
					x=[float(i) for i in x]
					inputs.append(np.array(x))
					classes.append(row[16])
				i+=1
    
	return np.array(inputs),np.array(classes)

def distance(row1,row2):
  dist=0
  for i in range(len(row1)):
    dist+=(float(row1[i])-float(row2[i]))**2
  if dist==0:
    return inf
  return sqrt(dist)

def knn(training_inputs,training_outputs,new_point,k=5):
  distances=[distance(training_inputs[i],new_point) for i in range(len(training_inputs))]
  if inf in distances:
        idx=distances.index(inf)
        return training_outputs[idx]
  neighbours=list(zip(distances[:k],training_outputs[:k]))
  #print("neighbours:",neighbours)
  neighbours.sort(key=lambda x:x[0],reverse=True)

  #print("len of training inputs v(max i value+1)",len(training_inputs))
  #print("max j value is:",k)
  for i in range(k,len(training_inputs)):
    j=0
    #print("i=",i)
    while distances[i]<neighbours[j][0]:
      j+=1
      if j!=0:
        neighbours.insert(j,(distances[i],training_outputs[i]))
        neighbours.pop(0)
      if j==k:
        break
      #print("j value is:",j)	

    one_count=0
    zero_count=0
    for n_neighbours in neighbours:
      if n_neighbours[1]=='0': 
        zero_count+=(1/n_neighbours[0])
      else:
        one_count+=(1/n_neighbours[0])
    
    if zero_count>one_count:
      return '0'
    return '1'

def upsample(input_data,output_data):
  l1=list(zip(input_data,output_data))
	minority=[]
	majority=[]
	for x in l1:
	  l2=[]
		l2=l2+x[0]+[x[1]]
		if l2[-1]==
  resample(minority)

#input_data4,output_data4=get_data('cat4.csv')
input_data3,output_data3=get_data('cat3.csv')
input_data2,output_data2=get_data('cat2.csv')
input_data1,output_data1=get_data('cat1.csv')



#get all majority, minority data poitns seperated
# resample(minority class,replace=True,len(majority_class))


#input_train4,input_test4,output_train4,output_test4=model_selection.train_test_split(input_data4,output_data4,test_size=0.6)
input_train3,input_test3,output_train3,output_test3=model_selection.train_test_split(input_data3,output_data3,test_size=0.6)
input_train2,input_test2,output_train2,output_test2=model_selection.train_test_split(input_data2,output_data2,test_size=0.6)
input_train1,input_test1,output_train1,output_test1=model_selection.train_test_split(input_data1,output_data1,test_size=0.6)

total=0
correct=0
tp0=0
fp0=0
tn0=0
fn0=0
tp1=0
fp1=0
tn1=0
fn1=0
print("calculating for cat1")
for i in range(len(input_test1)):
  ans=knn(input_train1,output_train1,input_test1[i],5)
  
  if ans==output_test1[i]:  #correct prediction
    correct+=1
    if output_test1[i]=='0':
      tp0+=1
      tn1+=1
    else:
      tp1+=1
      tn0+=1
  
  else:
    if output_test1[i]=='0': #wrong prediction
      fn0+=1
      fp1+=1
    else:
      fp0+=1
      fn1+=1    
  total+=1

precision0=tp0/(tp0+fp0)
precision1=tp1/(tp1+fp1)
recall0=tp0/(tp0+fn0)
recall1=tp1/(tp1+fn1)
print("accuracy for cat1 is:\n",correct/total)

print("precision for cat1 for class 0 is:",precision0)
print("recall for cat1 for class 0 is:",recall0)
print("f1 score for class0 is:",(2*precision0*recall0)/(precision0+recall0),"\n")

print("precision for cat1 for class 1 is:",precision1)
print("recall for cat1 for class 1 is:",recall1)
print("f1 score for class1 is:",(2*precision1*recall1)/(precision1+recall1),"\n")

total=0
correct=0
tp0=0
fp0=0
tn0=0
fn0=0
tp1=0
fp1=0
tn1=0
fn1=0
print("calculating for cat2")
for i in range(len(input_test2)):
  ans=knn(input_train2,output_train2,input_test2[i],5)
  
  if ans==output_test2[i]:  #correct prediction
    correct+=1
    if output_test2[i]=='0':
      tp0+=1
      tn1+=1
    else:
      tp1+=1
      tn0+=1
  
  else:
    if output_test2[i]=='0': #wrong prediction
      fn0+=1
      fp1+=1
    else:
      fp0+=1
      fn1+=1    
  total+=1

print("accuracy for cat2 is:\n",correct/total)

print("precision for cat2 for class 0 is:",precision0)
print("recall for cat2 for class 0 is:",recall0)
print("f1 score for class0 is:",(2*precision0*recall0)/(precision0+recall0),"\n")

print("precision for cat2 for class 1 is:",precision1)
print("recall for cat2 for class 1 is:",recall1)
print("f1 score for class1 is:",(2*precision1*recall1)/(precision1+recall1),"\n")

total=0
correct=0
tp0=0
fp0=0
tn0=0
fn0=0
tp1=0
fp1=0
tn1=0
fn1=0
print("calculating for cat3")
for i in range(len(input_test3)):
  ans=knn(input_train3,output_train3,input_test3[i],5)
  
  if ans==output_test3[i]:  #correct prediction
    correct+=1
    if output_test3[i]=='0':
      tp0+=1
      tn1+=1
    else:
      tp1+=1
      tn0+=1
  
  else:
    if output_test3[i]=='0': #wrong prediction
      fn0+=1
      fp1+=1
    else:
      fp0+=1
      fn1+=1    
  total+=1

print("accuracy for cat3 is:\n",correct/total)

print("precision for cat3 for class 0 is:",precision0)
print("recall for cat3 for class 0 is:",recall0)
print("f1 score for class0 is:",(2*precision0*recall0)/(precision0+recall0),"\n")

print("precision for cat3 for class 1 is:",precision1)
print("recall for cat3 for class 1 is:",recall1)
print("f1 score for class1 is:",(2*precision1*recall1)/(precision1+recall1),"\n")

