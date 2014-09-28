from snap import *
from pandas import *
import numpy as np
import statsmodels.api as sm
import pylab as pl
#from sklearn import *
from sklearn import cross_validation 
from  pickle import *
from sklearn.externals import joblib

triad_feature_X="triad_feature_X.txt"
triad_feature_Y="triad_feature_Y.txt"

#funcitons
def find_mutual_triad_node(node1, node2, graph):
	nbr_node1=[]
	nbr_node2=[]	
	for neighbor in node1.GetOutEdges():
		nbr_node1.append(neighbor)
	for neighbor in node2.GetOutEdges():
		nbr_node2.append(neighbor)

	
	intersect=[val for val in nbr_node1 if val in nbr_node2]
	return intersect

def polarity_training(polarity,graph,edge_sign):
	#*****************node polarity
	X=[]
	Y=[]
	for node1 in graph.Nodes():
		for nbr in node1.GetOutEdges():
			sign=edge_sign[(node1.GetId(),nbr.GetDstId())]
			if node1.GetId() in node_polarity:
				if sign==neg_sign:
					polarity[node1.GetId()]=-1
				else:
					polarity[node1.GetId()]=1
			else:
				if sign==neg_sign:
					polarity[node1.GetId()]+=-1
				else:	
					polarity[node1.GetId()]+=1
	for NI in graph.Nodes():
		for EI in NI.GetOutEdges():
			X.append([polarity(NI.GetId()),polarity(EI.GetDstId())])
			#X.append([EI.GetDstId(),NI.GetId()])
			Y.append(edge_sign[(NI.GetId(),EI.GetDstId())])
		

	#********train and save
	model=sm.GLM(y_vec,x_vec,family=sm.families.Binomial())
	result=model.fit()
	joblib.dump(result,'polarity_fitted_model100.pkl')
	#saved_model=joblib.load('fitted_model.pkl')

	K_fold(0.5,1,X,Y) #test_siz0=0.1

	return	
####
def traverse_triad(graph,edge_sign):
	x_vec=[]
	y_vec=[]
	total=0
	frac_triad=[0]*8
	for node1 in graph.Nodes():
		for node2 in graph.Nodes():

			#init triad features
			triad_4=[0,0,0,0]
			if node1.GetId()==node2.GetId():
				continue
			if not graph.IsEdge(node1.GetId(),node2.GetId()):
				continue
			intersect=find_mutual_triad_node(node1, node2, graph)
			
			for w in intersect:
				s_n1w=edge_sign[(node1.GetId(),w)]
				s_wn2=edge_sign[(w,node2.GetId())]
				'''if s_n1w==0:
					s_n1w=-1
				if s_wn2==0:
					s_wn2=-1'''
				ind=int(s_n1w*2+s_wn2)
				triad_4[ind]+=1
				frac_triad[int(ind*2+edge_sign[(node1.GetId(),node2.GetId())])]+=1
				total+=1
			x_vec.append(triad_4)	
			s_tmp=edge_sign[(node1.GetId(),node2.GetId())]
			y_vec.append(s_tmp)

	
	dump(x_vec,open("triad_feature_X","wb"))		
	dump(y_vec,open("triad_feature_Y","wb"))
	return x_vec,y_vec,total,frac_triad

def triad_training(graph,edge_sign):
	x_vec=[]
	y_vec=[]


	frac_triad=[0]*8 # 0 0 0, 0 0 1, 0 1 0
	total=0
	#traverse on each pair of node and find the number of each type of triangle
	# change!
	#x_vec,y_vec,total,frac_triad=traverse_triad(graph,edge_sign)		
	x_vec=load(open("triad_feature_X","rb"))
	y_vec=load(open("triad_feature_Y","rb"))
#	print x_vec,y_vec
	
	#find the fractaion of each triagle: 
#Train
	#change!
#************************cross validation
	print "cross_validation"
	t_size=0.9
	X_train, X_test, y_train, y_test=cross_validation.train_test_split(x_vec,y_vec,test_size=t_size, random_state=0)
	CV=sm.GLM(y_train,X_train,family=sm.families.Binomial())
	CV_fitted=CV.fit()
	test=CV_fitted.predict(X_test)

	error=0
	wrong_pos=0
	wrong_neg=0
	pos=0
	neg=0
	neg_pos=0
	real_pos=0
	real_neg=0
	right_pos=0
	right_neg=0
	#print test
	for i in range(len(y_test)):
		if test[i]>=0.5:
			pos+=1
		elif test[i]<0.5:
			neg+=1

		if y_test[i]<=0.5:
			real_neg+=1
		else:
			real_pos+=1
	
		if (test[i]-y_test[i])>=0.5:
			error+=1
			if y_test[i]>0.5:
				wrong_pos+=1
			else:
				wrong_neg+=1
		else:
			if y_test[i]>=0.5:
				right_pos+=1
			else:
				right_neg+=1
	print "pos", pos
	print "real pos",real_pos
	print "test_size: ", t_size
	print "error rate on the test set is:", error/float(len(y_test))
	print "positives precision(what fraction of + is correct): ",right_pos/float(pos)
	print "positive recall(what fraction of real +s it finds: ", right_pos/float(real_pos)
	print "negative precision(what fraction of - is correct): ",right_neg/float(neg)
	print "negative recall(what fraction of real -s it finds: ", right_neg/float(real_neg)


#	joblib.dump(CV_fitted,'fitted_90%.plk')
	return

def  K_fold(test_size,times,x_vec,y_vec): #test_siz0=0.1
	error=[]
	for i in range(times):

		X_train, X_test, y_train, y_test=cross_validation.train_test_split(x_vec,y_vec,test_size=test_size, random_state=0)
		CV=sm.GLM(y_train,X_test,family=sm.families.Binomial())
		CV_fitted=CV.fit()
		test=CV_fitted.predict(X_test)
		


		error[i]=0
		for i in range(len(y_test)):
			if (test[i]-y_test[i])>=0.5:
				error[i]+=1
	print "error rate on the test set is:", error
	print "average error rate is:",sum(error)/float(len(y_test))
	print "test_size", test_size,"times: ",times
	 
	#joblib.dump(CV_fitted,'fitted_90%.plk')
	return
########changed sign!
# define
pos_sign=1
neg_sign=0
attr="sign"
####

graph=TUNGraph.New()  #undirected graph
#inputF=open('input.txt','r')
#inputF=open('pos_neg.txt','r')	#converted 0 to 1
inputF=open('pos_neg_1M.txt')
edge_sign={}
polarity={}
#read graph
for line in inputF:
	count=0

	if len(line)==1: #it read one line with length 1 at the end! :|
		break
	for word in line.split('\t' or '\n'):
		
		if count==0:
			node1=int(word)
			count+=1
		elif count==1:
			node2=int(word)
			count+=1
		elif count==2:
			
			s=float(word) #signes are supposed to be -1 or 1
			#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			if s!=pos_sign:
				s=neg_sign
			
			count=0
			if not graph.IsNode(node1):
				graph.AddNode(node1)
			if not graph.IsNode(node2):
				graph.AddNode(node2)
			graph.AddEdge(node1,node2)
			#print "add edge", node1, node2
			edge_sign[(node1,node2)]=s
			edge_sign[(node2,node1)]=s

			#graph.AddIntAttrDatE(edge_id,s,attr) #!!!assumed edge id starts from 0!
inputF.close()

#number of triads
triad_num= GetTriads(graph)
print "triad count: ", triad_num
print "number of nodes: ",graph.GetNodes()
print "number of edges: ",graph.GetEdges()

triad_training(graph,edge_sign)
polarity={}
#polarity_training(polarity,graph,edge_sign)
