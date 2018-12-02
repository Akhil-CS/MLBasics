import numpy as np
import random
import matplotlib.pyplot as mpl


list3=[]
weights=[]
dot_product=[]
w=[]

	
list3=np.loadtxt("iris_data_binary.txt")
np.random.shuffle(list3)


dataset=[45,60,75,90,105]


for train_no in dataset:
	weights=(np.random.rand(1,4))[0]

	wrong_classified=0
	misclassified=0

	
	error=[]
	error_out=[]
	best_weights=weights

	misclassified=0
	max_score=0
	score=0
	for n in range(10):
		misclassified=0
		for i in list3[0:train_no]:
			
			Error=0
			dot_product=np.dot(i[0:4],weights)
			error_iterate=np.absolute((dot_product-i[4]))
			if dot_product<0:
				sign=-1
			
			else:
				sign=1
			
		
		
			if sign==i[4]:
				score=score+1
				
	
			else:
				if score>max_score:
					max_score=score
					best_weights=weights
				weights += i[4]*i[0:4]
				misclassified += 1
				score=0
		error.append(misclassified)	
		
		wrong_classified=0
		for i in list3[106:]:
			Error=0
			dot_product=np.dot(i[0:4],best_weights)
			if dot_product<0:
				sign=-1
			
			else:
				sign=1
			if sign!=i[4]:
				wrong_classified += 1;
							
		error_out.append(wrong_classified) 
			

	print(best_weights)
	xs=range(10)
	ys=[error[x] for x in xs]
	mpl.title(train_no)
	mpl.xlabel("Iterations for pocket")
	mpl.ylabel("Error")
	mpl.plot(xs,ys)


	xs=range(10)
	ys=[error_out[x] for x in xs]
	print(error_out)
	mpl.title(train_no)
	mpl.xlabel("Iterations for pocket")
	mpl.ylabel("Error")
	mpl.plot(xs,ys)
	
	mpl.show()


	
