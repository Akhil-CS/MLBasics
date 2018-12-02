import numpy as np
import matplotlib.pyplot as mpl


data_file=open("winequality-white.csv","r")

list3=[]
list4=[]
weights=[]
dot_product=[]
pseudo_inverse=[]


data_set=[1200,1600,2000,2400,2800]
list3=np.loadtxt("winequality-white.csv",delimiter=';')
list4=list3[:,:11]
'''list4[:,:11]=np.square(list4[:,:11])'''

sq_list=[]
sq_list=np.square(list4[:,:11])
list4=np.concatenate((list4,sq_list),axis=1)

for i in range(11):
	for j in range(i+1,11):

		mul_list=list3[:,i]*list3[:,j]
		list4=np.concatenate((list4,np.array([mul_list]).T),axis=1)
'''print("l3 sh",list3.shape)
print(list4.shape)'''
a=np.ones((4898,1))
list4=np.concatenate((list4,a),axis=1)
print('l4i',list4.shape)

list_out=[]
for n in data_set:
	pseudo_inverse=np.linalg.pinv(list4[:n,:])
	#print((np.array(pseudo_inverse).T).shape)
	
	weights=np.dot(pseudo_inverse,list3[:n,11])
#	print(weights.shape)
	#print(weights)
	#E_in=0
	#list_in=[]
	

#training data

	'''for i in list4[0:n]:
		E_in=E_in+np.square(np.dot(np.transpose(weights),i[0:11])-i[11])

	E_in=E_in/n
	print(n,E_in)
	list_in.append(E_in)'''

	E_out=0
	x=list4[4500:,:]
	y=list3[:,11]
	
	for i in range(len(x)):
		E_out=E_out+np.square(np.dot(np.transpose(weights),x[i])-y[i])
		
	E_out=E_out/(4898-4500)
	print(n,E_out)
	list_out.append(E_out)
	
	

'''mpl.plot([n for n in data_set],list_out)
mpl.title("Error(out) in non-linear regression")
mpl.xlabel("data_size")
mpl.ylabel("error")
mpl.grid()
mpl.show()	
'''



