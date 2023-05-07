Download Link: https://assignmentchef.com/product/solved-ist-597-foundations-of-deep-learning-assignment-00010
<br>
<ul>

 <li><strong>Problem 1: Linear Regression                                                                                                                  </strong>(1 + 1 + 1 = 3 points)</li>

</ul>

Linear Regression To get you to think in terms of neural architectures, we will approach the problem of estimating good regression models from the perspective of incremental learning. In other words, instead of using the normal equations to directly solve for a closed-form solution, we will search for optima (particularly minima) by iteratively calculating partial derivatives (of some cost function with respect to parameters) and taking steps down the resultant error landscape. The ideas you will develop and implement in this assignment will apply to learning the parameters of more complex computation graphs, including those that define convolutional neural networks and recurrent neural net.

We create a complete example of using linear regression to predict the paramters of the function

<em>f</em>(<em>x</em>) = 3<em>x </em>+ 2 + <em>noise</em>

Given a point <em>x </em>we want to predict the value of <em>y</em>. We train the model on 10000 data pairs (<em>x,f</em>(<em>x</em>)). The model to learn is a linear model

<em>y</em>ˆ = <em>Wx </em>+ <em>b</em>

Note that, we use ‘tf.GradientTape‘ to record the gradient with respect our trainable paramters. We use Mean Square Error(MSE) to calcuate the loss

<em>g </em>= (<em>y </em>− <em>y</em>ˆ)<sup>2</sup>

Other loss function which can be used for eg L1

<em>g </em>= (<em>y </em>− <em>y</em>ˆ) We use Gradient Descent to update the paramters

1

<em>– Assignment #00010                                                                                                                                                                             </em>2

<h1>Things to Report</h1>

<strong>NOTE</strong>:- All submissions should use NIPS latex template.

Pdf generated from NIPS template would only be accepted rest all would lead to zero points.

<ul>

 <li>Fork repo [<em>https </em>: <em>//github.com/AnkurMali/IST</em>597 <em>Fall</em>2019 <em>TF</em>2<em>.</em>0] and modify file lin reg.py.</li>

 <li>Change the loss function,which loss function works better and why?Write mathematical formulation for each loss function.</li>

 <li>Create hybrid loss function(For eg. L1 + L2)</li>

 <li>Change the learning rate.</li>

 <li>Use patience scheduling[Whenever loss do not change , divide the learning rate by half].</li>

 <li>Train for longer Duration. Change the initial value for W and B.What effect it has on end result?</li>

 <li>Change the level of noise. • Use various type of noise.</li>

 <li>Add noise in data.</li>

 <li>Add noise in your weights.</li>

 <li>Add noise in your learning rate[For all above: Scheme can be per epoch or per N epochs]</li>

 <li>How do these changes effect the performance?</li>

 <li>Do you think these changes will have the same effect (if any) on other classification problems and mathematical models?</li>

 <li>Plot the different result.</li>

 <li>Do you get the exact same results if you run the Notebook multiple times without changing any parameters? Why or why not?[Explain significance of seed].</li>

 <li>Use unique seed for each experiment[Note:- Convert your first name into decimal].</li>

 <li>Later report per epoch GPU vs CPU Time.</li>

 <li>Can you get an model which is robust to noise?Does model lead to faster convergence?Do you get better local minima?Is noise beneficial?</li>

 <li>Collect everything and report your findings.</li>

</ul>

<strong>Problem 2: Logistic Regression                                                                                                                     </strong>(2+2+1=5 points)

Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables. In this you will be using Fashion Mnist which is a dataset of Zalando’s article imagesconsisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28×28 grayscale image, associated with a label from 10 classe[https://github.com/zalandoresearch/fashion-mnist].FashionMnist is much harder than MNIST so getting accuracy in 90’s is difficult. You can use whatever loss functions, optimizers, even models that you want, as long as your model is built in TensorFlow using eager execution[Remember no keras is allowed]

<h1>Things to Report</h1>

<ul>

 <li>Fork repo [https://github.com/AnkurMali/IST597 Fall2019 TF2.0] and modify file log reg.py.</li>

 <li>TODO: This keyword means you have to implement specific section/function/formula.</li>

 <li>Report should contain matplotlib plots from function plot images and plot weights.</li>

 <li>Change the optimizer and report which one converges faster and which one reaches better local minima/generalizes better.[Now you can use tensorflow optimizer,but no keras]</li>

</ul>

<em>– Assignment #00010                                                                                                                                                                             </em>3

<ul>

 <li>Train for longer epochs.</li>

 <li>Change Train/Val split.Report if you observe any performance drop/gain.</li>

 <li>Report Train/Val accuracy over time.</li>

 <li>Does batch size have any effect on performance.</li>

 <li>Report GPU vs CPU per epoch performance.</li>

 <li>Do Model overfit?If so why and also report measures you took to avoid overfitting.</li>

 <li>Compare performance with random forest and svm(you can use any built in library for only this)</li>

 <li>Bonus points[Cluster the weights for each class using any clustering mechanism].Show t-sne or k-means clusters.</li>

</ul>