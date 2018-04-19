# experimenting with neural network (FFNN) architecture
- inspired by courses of AndrewNg 
	- vectorization of batches
	- universal layered approach
- and udacity template
	- class encapsulation approach

- architecture is encapsulated more over to FeatureSpaces
	- each space has own weights & activation fuction
	- define derivative, back-prop and forward-pass on this level
- NeuralNetwork class is responsible only for puting it all together
	- initializing spaces
	- interaction with correct spaces during forward and backward prop