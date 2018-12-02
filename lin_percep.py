import numpy as np
import matplotlib.pyplot as mpl

list3=[]
weights=[]
dot_product=[]
w=[]
iterations = 10
learning_rate = 0.7
dataset=[45,60,75,90,105]
list3=np.loadtxt("iris_data_binary.txt")
np.random.shuffle(list3)

for train_no in dataset:
	weights=(np.random.rand(1,4))[0]
	
	wcld=0
	miscld=0

	#training data ,run multiple times
	#E(in) training data
	#E(out) testing data
	error_out=[]
	error=[]

	for n in range(iterations):
		w=weights
		miscld=0
		for i in list3[0:train_no]:
			dot_product=np.dot(i[0:4],weights)			
			if dot_product<0:
				sign=-1
			else:
				sign=1

			if sign!=i[4]:
				weights += learning_rate*i[4]*i[0:4]
				miscld += 1
		error.append(miscld)

	#testing data

		wcld=0
		for i in list3[106:]:
			dot_product=np.dot(i[0:4],weights)
			if dot_product<0:
				sign=-1
		
			else:
				sign=1
			if sign!=i[4]:
				wcld+=1;
		error_out.append(wcld) 

	print(error)
	print(error_out)	
	xs=range(iterations)
	ys=[error[x] for x in xs]
	mpl.title(train_no )
	mpl.xlabel("Iterations for pla")
	mpl.ylabel("No. of miscld points")
	mpl.plot(xs,ys)

	xs=range(iterations)
	ys=[error_out[x] for x in xs]
	mpl.title(train_no)
	mpl.xlabel("Iterations for pla")
	mpl.ylabel("No. of miscld points")
	mpl.plot(xs,ys)
	mpl.show()