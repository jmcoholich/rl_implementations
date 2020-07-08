#test of gradient tape
import torch
import time
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

for i in range(100):
	time.sleep(0.05)
	writer.add_scalar('ppheadongodnocaponehunned',i**2,i)
writer.close()

# x = torch.ones(1, requires_grad=True)
# x = torch.tensor([[1.,1],[1,1]], requires_grad=True)
# y = x + 2
# z = y * y * 2
# print(x)
# print(y)
# print(z)

# # print(torch.autograd.grad(z,x))
# z.backward(torch.ones(z.size()))     # automatically calculates the gradient
# print(x.grad)    

# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# print()
# tf.random.set_seed(2020)

# x = tf.Variable([[1.0,2.0]])

# model = Sequential()
# model.add(Dense(2,input_dim=2))

# with tf.GradientTape() as tape:
# 	# y = x**3
# 	y = model(x)
# 	print()
# 	print(y)
# 	temp = y[0,0]
# 	print(temp)


# print()
# print(model.layers[0].get_weights())
# print()
# print('trainable weights', model.trainable_weights)
# print()
# print('gradient',tape.gradient(temp,x))