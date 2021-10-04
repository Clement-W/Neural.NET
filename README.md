<br />

<h1 align="left">Neural .NET</h1>



## About The Project

What's the best way to fully understand something ? Doing it from sratch ! 
That's why I've implemented a little neural net library with automatic differentiation, inspired by Pytorch and Keras. This project was conducted for educational purposes and is not finished yet. Performances cannot compete with other c# neural network library like [NeuralNetwork.NET](https://github.com/Sergio0694/NeuralNetwork.NET). 

Great ressources that helped me a lot to carry out this project : 
* [3Blue1Brown](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) : Really good math videos that helped me to get the intuition of how neural network works.
* [Joel Grus](https://youtube.com/playlist?list=PLeDtc0GP5ICldMkRg-DkhpFX1rRBNHTCs) : A software engineer, data scientist, author who create great educational content. His series "livecoding an autograd library" helped me a lot to understand and implement the automatic differenciation part of this projet.
* [Deepmath - Exo7](https://exo7math.github.io/deepmath-exo7/) : A french book that explain the math behind neural networks very well. This is an awesome project carried out by Arnaud Bodin and François Recher, from Université de Lille 1.


### Built With

* [Dotnet 5.0](https://dotnet.microsoft.com/)
* [xUnit 2.4.1](https://xunit.net/)

## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

* Dotnet  ⩾ 5.0


and/or


* Visual Studio 2019

### Installation


1. Clone the repo
   ```sh
   git clone https://github.com/Clement-W/Neural.NET.git
   cd Neural.NET/DNN/
   ```
2. Open the solution DNN.sln with visual studio or visual studio code with .NET core and solution explorer extensions.

## Usage

### Create a model 

* With the sequential class :
```cs
Sequential model = new Sequential(
                new LinearLayer(2, 4),
                new LeakyRelu(),
                new LinearLayer(4, 8),
                new LeakyRelu(),
                new LinearLayer(8, 2),
                new Sigmoid()
            );
```

* Or by inheriting from the Model class :

```cs
class MyModel : Model
    {
        public LinearLayer Linear1 { get; set; }
        public LinearLayer Linear2 { get; set; }
        public LinearLayer Linear3 { get; set; }

        public LeakyRelu Activation1 { get; set; }
        public LeakyRelu Activation2 { get; set; }
        public Sigmoid Activation3 { get; set; }


        public MyModel()
        {
            Linear1 = new LinearLayer(2, 5);
            Activation1 = new LeakyRelu();
            Linear2 = new LinearLayer(5, 5);
            Activation2 = new LeakyRelu();
            Linear3 = new LinearLayer(5, 2);
            Activation3 = new Sigmoid();

        }

        public override Tensor Predict(Tensor inputs)
        {
            Tensor output;
            output = Linear1.Forward(inputs);
            output = Activation1.Forward(output);
            output = Linear2.Forward(output);
            output = Activation2.Forward(output);
            output = Linear3.Forward(output);
            output = Activation3.Forward(output);
            return output;
        }
    }
```
### Compile the model 

Compile the model with an optimizer and a loss function :
```cs
Optimizer optimizer = new SGD(lr: 0.03);
ILoss mse = new MSE();
model.Compile(optimizer,mse);
```

### Load the data 

The Dataloader class will split the data in multiple batches :
```cs
Tensor xData = new Tensor(requiresGrad: true, shape: new int[] { 4, 2 }, 0, 0, 1, 0, 0, 1, 1, 1);
Tensor yData = new Tensor(requiresGrad: true, shape: new int[] { 4, 2 }, 1, 0, 0, 1, 0, 1, 1, 0);

int batchSize = 2;
DataLoader trainData = new DataLoader(xData, yData, batchSize,shuffle: true);
```

### Train the model

```cs
int nbEpochs = 500;
model.Train(trainData, nbEpochs, verbose: true);
```

### Evaluate the model

```cs
model.Evaluate(testData);
```

## Demo

You can find more in the two examples I've implemented to test the library : 
* Xor : Simple neural net that learned the xor operation.
* Circles classification : Binary classification of 2 group of data points in circles.

## Testing

With xUnit 2.4.1, use ```dotnet test``` to execute the unit tests.


## Contributing

I'm still a beginner in this field, so feel free to use Issues or PR to report errors and/or propose additions or corrections to my code. Any contributions you make are greatly appreciated !

## License

Distributed under the MIT License. See `LICENSE` for more information.


## Todo list

- [ ] Add softmax
- [ ] Add multi class classification
- [ ] Add momentum for sgd
- [ ] Add callbacks, reduce lr on plateau
- [ ] Add support for >2dim array for NDimArray by generalizing those methods: Sum, Slice2DArrays, ApplyBroadcastOperationBetween2DArrays, Extend2DArrayByShape ,Transpose, Max (GetPredictionsIndexes)

