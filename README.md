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

# Example of usage
```python
class NeuralNetwork(zer0nn.NeuralNetwork):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        super(NeuralNetwork, self).__init__(
                [(None, input_nodes), ("sig", hidden_nodes), ("lin", output_nodes)],
                learning_rate,
                .995)
```
