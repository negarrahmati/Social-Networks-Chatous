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
	for neighbor in (graph.GetNI(node1)).GetOutEdges():
		nbr_node1.append(neighbor)
	for neighbor in (graph.GetNI(node2)).GetOutEdges():
		nbr_node2.append(neighbor)

	
	intersect=[val for val in nbr_node1 if val in nbr_node2]
	return intersect

def polarity_training(polarity,graph,edge_sign,removed_edges,removed_edge_enemy):
	#*****************node polarity
	#not implemented! feature vec should be: deg+ u, deg- u, deg+ v, deg- v,deg u, deg v, embededness (it should be tested for different levels of embeddedness
	pos_deg={}
	neg_deg={}
	x_vec=[]
	y_vec=[]
	#get features for polarity training
		
	for node1 in graph.Nodes():
		degn=0
		degp=0
		for nbr in node1.GetOutEdges():
			sign=edge_sign[(node1.GetId(),nbr)]
			if sign==neg_sign:
				degn+=1
			else:
				degp+=1
		polarity[ node1.GetId()]=[degp,degn,degp+degn]	
		#print node1, polarity[node1]
	for EI in graph.Edges():
		#print "!"
		tmp=[]
		u=EI.GetSrcNId()
		v=EI.GetDstNId()
		tmp=polarity[u]+polarity[v]+[(len(find_mutual_triad_node(graph.GetNI(u),graph.GetNI(v),graph)))]
		#tmp=polarity[u]+polarity[v]+[0]
		x_vec.append(tmp)
		y_vec.append(edge_sign[(u,v)])
	#print x_vec
	#print y_vec
	
	#********train and save
	dump(x_vec,open("feature_len_polarity_removed_friends_X","wb"))		
	dump(y_vec,open("feature_len_polarity_removed_friends_Y","wb"))
	print "saved features for polarity training, 10 percent of friends removed"	
	x_vec_f=[]
	y_vec_f=[]
	for (u,v) in removed_edges:
		#print "!"
		tmp=[]
		tmp=polarity[u]+polarity[v]
		tmp.append(len(find_mutual_triad_node((u),(v),graph)))
		x_vec_f.append(tmp)
		y_vec_f.append(pos_sign)

	for (u,v) in removed_edge_enemy:
		#print "!"
		tmp=[]
		tmp=polarity[u]+polarity[v]
		tmp.append(len(find_mutual_triad_node((u),(v),graph)))
		x_vec_f.append(tmp)
		y_vec_f.append(neg_sign)
	
	dump(x_vec_f,open("feature_len_polarity_friends_X","wb"))		
	dump(y_vec_f,open("feature_len_polarity_friends_Y","wb"))
	print "saved features for polarity training of removed friends, 10 percent of friends removed"	
	'''
	x_vec=load(open("feature_polarity_removed_friends_X","rb"))
	y_vec=load(open("feature_polarity_removed_friends_Y","rb"))
	x_vec_f=load(open("feature_polarity_friends_X","rb"))
	y_vec_f=load(open("feature_polarity_friends_Y","rb"))
	print "!!!",len(x_vec_f)
	#print x_vec_f
	#print x_vec
	#print len(x_vec)
	#print len(y_vec)
	#model=sm.GLM(y_vec,x_vec,family=sm.families.Binomial())
	#result=model.fit()
	#joblib.dump(result,'polarity_fitted_model100.pkl')
	##saved_model=joblib.load('fitted_model.pkl')
	#K_fold(0.1,1,x_vec,y_vec) #test_siz0=0.1
	'''
	###Cross validation
	print "cross_validation"
	#print removed_edge_enemy
	t_size=0.1
	#X_train, X_test, y_train, y_test=cross_validation.train_test_split(x_vec,y_vec,test_size=t_size, random_state=0)
	X_train=x_vec
	X_test=x_vec_f
	y_train=y_vec
	y_test=y_vec_f
	
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

		if y_test[i]<0.5:
			real_neg+=1
		else:
			real_pos+=1
	
		if (test[i]-y_test[i])>=0.5 or (test[i]-y_test[i]<(-0.5)):
			error+=1
			if y_test[i]>=0.5:
				wrong_pos+=1
			else:
				wrong_neg+=1
		else:
			if y_test[i]>=0.5:
				right_pos+=1
			else:
				right_neg+=1
	print "pos", pos
	print "right_pos",right_pos
	print "real pos",real_pos
	print "test_size: ", t_size
	print "error rate on the test set is:", error/float(len(y_test))
	print "positives precision(what fraction of + is correct): ",right_pos/float(pos)
	print "positive recall(what fraction of real +s it finds: ", right_pos/float(real_pos)
	if neg!=0:
		print "negative precision(what fraction of - is correct): ",right_neg/float(neg)
	if real_neg!=0:
		print "negative recall(what fraction of real -s it finds: ", right_neg/float(real_neg)

	return	
####
def traverse_triad(graph,edge_sign,removed_edges,removed_edge_enemy):
	frac_triad=[0]*8
	x_vec=[]
	y_vec=[]
	total=0
	dummy=0
	for EI in graph.Edges():
		dummy+=1
		node1=EI.GetSrcNId()
		node2=EI.GetDstNId()	
		if (dummy==10000):
			print dummy
		#print (node1.GetId(), node2.GetId())
		#init triad features
		triad_4=[0,0,0,0]
		if node1==node2:
			continue
		if not graph.IsEdge(node1,node2):
			continue
		intersect=find_mutual_triad_node(node1, node2, graph)
		## only edges included in a traingle are trained!
		## watch out!
		if len(intersect)==0:
			continue
		for w in intersect:
			s_n1w=edge_sign[(node1,w)]
			s_wn2=edge_sign[(w,node2)]
			ind=int(s_n1w*2+s_wn2)
			triad_4[ind]+=1
			frac_triad[int(ind*2+edge_sign[(node1,node2)])]+=1
			total+=1
		x_vec.append(triad_4+[len(intersect)])	
		s_tmp=edge_sign[(node1,node2)]
		y_vec.append(s_tmp)

	dump(x_vec,open("feature_len_triad_removed_friend_X","wb"))		
	dump(y_vec,open("feature_len_triad_removed_friend_Y","wb"))	
	print "saved features for triad training, 10 percent of friends removed"	
	
	#x_vec=load(open("feature_triad_removed_friend_X","rb"))
	#y_vec=load(open("feature_triad_removed_friend_Y","rb"))
	print "loaded features of the network when the 10 percent of friends are removed"

	x_vec_f=[]
	y_vec_f=[]
	count_ignored=0
	#print "removed friends",removed_edges
	#print "enemy",removed_edge_enemy
	for (node1,node2) in removed_edges:
		triad_4=[0,0,0,0]
		if node1==node2:
			continue
		'''	if not graph.IsEdge(node1,node2):
			continue
		'''
		intersect=find_mutual_triad_node((node1),(node2), graph)
		'''if (len(intersect)==0):
			count_ignored+=1
			continue	
		'''
		for w in intersect:
			s_n1w=edge_sign[(node1,w)]
			s_wn2=edge_sign[(w,node2)]
			'''if s_n1w==0:
				s_n1w=-1
			if s_wn2==0:
				s_wn2=-1'''
			ind=int(s_n1w*2+s_wn2)
			triad_4[ind]+=1
			#frac_triad[int(ind*2+edge_sign[(node1,node2)])]+=1
			total+=1
		x_vec_f.append(triad_4+[len(intersect)])	
		#s_tmp=edge_sign[(node1,node2)]
		y_vec_f.append(pos_sign)
	
	for (node1,node2) in removed_edge_enemy:
		triad_4=[0,0,0,0]
		if node1==node2:
			continue
		'''if not graph.IsEdge(node1,node2):
			continue
		'''
		intersect=find_mutual_triad_node((node1),(node2), graph)
		"""if (len(intersect)==0):
			count_ignored+=1
			continue	
		"""
		for w in intersect:
			s_n1w=edge_sign[(node1,w)]
			s_wn2=edge_sign[(w,node2)]
			""""if s_n1w==0:
				s_n1w=-1
			if s_wn2==0:
				s_wn2=-1"""
			ind=int(s_n1w*2+s_wn2)
			triad_4[ind]+=1
			#frac_triad[int(ind*2+edge_sign[(node1,node2)])]+=1
			total+=1
		x_vec_f.append(triad_4+[len(intersect)])	
		#s_tmp=edge_sign[(node1,node2)]
		y_vec_f.append(neg_sign)
	dump(x_vec_f,open("feature_len_triad_friend_X","wb"))		
	dump(y_vec_f,open("feature_len_triad_friend_Y","wb"))
	print "saved features for triad training of removed friends(secod phase), 10 percent of friends removed"	
	dump(frac_triad,open("frac_triad_friends_testntrain","wb"))
	print "ignored edges in getting features of the test set: ",count_ignored
	return x_vec,y_vec,x_vec_f,y_vec_f,total,frac_triad

def triad_training(graph,edge_sign,removed_edges,removed_edge_enemy):
	x_vec=[]
	y_vec=[]
	x_vec_f=[]
	y_vec_f=[]

	frac_triad=[0]*8 # 0 0 0, 0 0 1, 0 1 0
	total=0
	#traverse on each pair of node and find the number of each type of triangle
	# change!
	x_vec,y_vec,x_vec_f,y_vec_f,total,frac_triad=traverse_triad(graph,edge_sign,removed_edges,removed_edge_enemy)		
	#x_vec=load(open("feature_triad_removed_friend_X","rb"))
	#y_vec=load(open("feature_triad_removed_friend_Y","rb"))
	#x_vec_f=load(open("feature_triad_friend_X","rb"))
	#y_vec_f=load(open("feature_triad_friend_Y","rb"))
	print "!!!!",len(x_vec_f)
	print len(y_vec)
	#print x_vec_f
	#find the fractaion of each triagle: 
	#Train
	#change!
#************************cross validation
	print "cross_validation"
	#print removed_edge_enemy
	t_size=0.1
	#X_train, X_test, y_train, y_test=cross_validation.train_test_split(x_vec,y_vec,test_size=t_size, random_state=0)
	X_train=x_vec
	X_test=x_vec_f
	y_train=y_vec
	y_test=y_vec_f
	print "LEN",len(y_vec),len(x_vec)
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

		if y_test[i]<0.5:
			real_neg+=1
		else:
			real_pos+=1
	
		if (test[i]-y_test[i])>=0.5 or (test[i]-y_test[i]<(-0.5)):
			error+=1
			if y_test[i]>=0.5:
				wrong_pos+=1
			else:
				wrong_neg+=1
		else:
			if y_test[i]>=0.5:
				right_pos+=1
			else:
				right_neg+=1
	print "pos", pos
	print "right_pos",right_pos
	print "real pos",real_pos
	print "test_size: ", t_size
	print "error rate on the test set is:", error/float(len(y_test))
	print "positives precision(what fraction of + is correct): ",right_pos/float(pos)
	print "positive recall(what fraction of real +s it finds: ", right_pos/float(real_pos)
	if neg!=0:
		print "negative precision(what fraction of - is correct): ",right_neg/float(neg)
	if real_neg!=0:
		print "negative recall(what fraction of real -s it finds: ", right_neg/float(real_neg)
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
#inputF=open('pos_neg_1M.txt')
#inputF=open('pos_neg_naiveBayes_500k.txt')
#inputF=open('pos_neg_naiveBayes_1M.txt')
inputF=open('pos_neg_len_all.txt')
inputFriends=open('pos_neg_gold (1).txt')
friend={}
enemy={}
er_count=0
en_count=0
line_count=0
for line in inputFriends:
	#print line
	count=0
	if len(line)==1: #it read one line with length 1 at the end! :|
		er_count+=1
		break
	for word in line.split('\t' or '\n'):
		
		if count==0:
			node1=int(word)
			count+=1
		elif count==1:
			node2=int(word)
			count+=1
		elif count==2:
			sign=float(word)
	#print sign
	if int(sign)==pos_sign:
		friend[(node1,node2)]=1
		friend[(node2,node1)]=1

	else:	
		enemy[(node1,node2)]=1
		enemy[(node2,node1)]=1
	
	
#print "enemy count", en_count
#print "er count",er_count
edge_sign={}
polarity={}
#read graph
frac_f=0
base_test=50
frac_e=0
removed_edge=[]
removed_edge_enemy=[]
for line in inputF:
	if line_count==1000000:
		break
	line_count+=1
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
			if (node1,node2) in friend:
				#print "friend found"
				frac_f+=1
				if frac_f==base_test:
					frac_f=0
					removed_edge.append((node1,node2))
				else:	
					graph.AddEdge(node1,node2)
					
					#print "add edge", node1, node2
					edge_sign[(node1,node2)]=s
					edge_sign[(node2,node1)]=s
			if (node1,node2) in enemy:
				#print "enemy found!"
				frac_e+=1
				if frac_e==base_test:
					frac_e=0
					removed_edge_enemy.append((node1,node2))
				else:	
					graph.AddEdge(node1,node2)
					
					#print "add edge", node1, node2
					edge_sign[(node1,node2)]=s
					edge_sign[(node2,node1)]=s
			
			#graph.AddIntAttrDatE(edge_id,s,attr) #!!!assumed edge id starts from 0!
#print removed_edge
print "len removed friend", len(removed_edge)
print "len friends", len(friend)/2
print "len removed enemy",len(removed_edge_enemy)
print "len enemy",len(enemy)/2

inputF.close()
inputFriends.close()
#number of triads
triad_num= GetTriads(graph)
print "triad count: ", triad_num
print "number of nodes: ",graph.GetNodes()
print "number of edges: ",graph.GetEdges()
print "####triad: "
triad_training(graph,edge_sign,removed_edge,removed_edge_enemy)
polarity={}
print "####polarity"
polarity_training(polarity,graph,edge_sign,removed_edge,removed_edge_enemy)
