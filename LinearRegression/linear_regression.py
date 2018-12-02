import numpy as np
import matplotlib.pyplot as mpl

list3=[]
weights=[]
dot_product=[]
pseudo_inverse=[]


data_set=[1200,1600,2000,2400,2800]
list3=np.loadtxt("winequality-white.csv",delimiter=';')
list_out=[]
for n in data_set:
	pseudo_inverse=np.linalg.pinv(list3[:n ,:11])

	weights=np.dot(pseudo_inverse,list3[:n ,11])
	print("weights")
	print(weights)
	E_in=0
	list_in=[]
	
	E_out=0

	for i in list3[4001:]:
		E_out=E_out+np.square(np.dot(np.transpose(weights),i[0:11])-i[11])

	E_out=E_out/899
	print(n,E_out)
	list_out.append(E_out)


mpl.plot([n for n in data_set],list_out)
mpl.title("E_out linear regression")
mpl.xlabel("data_set")
mpl.ylabel("error")
mpl.show()	




