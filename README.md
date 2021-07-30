<br />

<h1 align="left">Neural .NET</h1>



## About The Project

What's the best way to fully understand something ? Doing it from sratch ! 
That's why i've implemented a little neural net library with automatic differenciation, inspired by Pytorch and Keras. This project was conducted for educational purposes and is not complete at all. Performances cannot compete with other c# neural network library like [NeuralNetwork.NET](https://github.com/Sergio0694/NeuralNetwork.NET). I have chosen to use c# over python because most of the tensors operation are already implemented with numpy.

Great ressources that helped me a lot to carry out this project : 
* [3Blue1Brown](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) : Really good math videos that helped me to get the intuition of how neural network work.
* [Joel Grus](https://youtube.com/playlist?list=PLeDtc0GP5ICldMkRg-DkhpFX1rRBNHTCs) : A software engineer, data scientist, author who create great educational content. His series "livecoding an autograd library" helped me a lot to understand and implement the automatic differenciation part of this projet.
* [Deepmath - Exo7](https://exo7math.github.io/deepmath-exo7/) : A french book that explain the math behind neural networks. very well. This is an awesome project carried out by Arnaud Bodin and François Recher, from Université de Lille 1.
*

### Built With

* [Dotnet 5.0.301](https://dotnet.microsoft.com/)


## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

* Dotnet  ⩾ 5.0
and/or
* Visual Studio 2019

### Installation


1. Clone the repo
   ```sh
   git clone https://github.com/Clement-W/DNN-Autodiff-From-Scratch.git
   cd DNN-Autodiff-From-Scratch/DNN/
   ```
2. Open the solution DNN.sln with visual studio or visual studio code with .NET core and solution explorer extensions.


## Usage


## Demo


blablabla:
```py
blabla
```


<!-- CONTRIBUTING -->
## Contributing

I'm still a beginner in this field, so feel free to use Issues or PR to report errors and/or propose additions or corrections to my code. Any contributions you make are greatly appreciated !


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- TODO LIST -->
## Todo list

- [ ] softmax
- [ ] multi class classification
- [ ] sgd momentum
- [ ] callbacks, reduce lr on plateau
- [ ] Improve ndimarray and tensors efficiency
