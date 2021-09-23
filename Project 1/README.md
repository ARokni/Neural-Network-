# First Project
 
 * In this project, I have implemented a **`Multi-Category Image Classification`** task via **`CNN`** models on **`German Traffic Sign Recognition Benchmark`** dataset utilizing **`Keras`**.

 * The project focused mainly on the following topics:

    -   Developing a **CNN** model for the classification task

    -   Investigation the effect of various **`Activation Functions`**:**`ReLU`**, **`tanh`**, and **`Sigmoid`**,  on the final model's performance.

    - Investigation the effect of various **`Optimizers`**: **`Gradient Descent`** and **`Adam`**, on the final model's performance.
    
    - Investigation the effect of **`Dropout`** and **`Batch Normalization`** on the final model's performance.

    - Investigation the effect of **`Data Augmentaton`** on the final model's performance.

    - Providing **`Loss Accuracy`** for **`Validation`** and **`Test`** data and depicting **`Confusion Matrix`** for the profound assement of the ability of the model in prediction of each seperate class.

* Notble Notes:
    -  On the ground of [[1]](#1), I have expected **ReLU** activation outperform **Sigmoid** and **tanh** since ReLU does not saturate and is computationally efficient. Moreover,  ReLu converges much fsate than Sigmoid/tanh in practice(e.g. 6X). By simulation I figured out the results are consistent with the reference paper.

    -  Based on [[2]](#2) , we know that **Gradient Descent** optimizer outperfroms **Adaptive** methods in **Image Processing** tasks; however, I applied **Adam** since adaptive optimizers have a higher convergence rate.

    - In **Data Augmentation** I found that **rotation** results in poor performance of the final model. In fact, since some different **Traffic Signs** are symmetric, applying the rotation results in amiguity and misunderstaing in models. As a result I eliminated the roation transformation.

* The codes are provided in [Multi-Object Classification.py](https://github.com/ARokni/Neural-Network-/blob/main/Project%201/Multi-Object%20Classification.py).



    




## References
<a id="1">[1]</a> 
[Krizhevsky, A., Sutskever, I. and Hinton, G.E., 2012. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, pp.1097-1105.](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

<a id="2">[2]</a> 
[Zhang, J., Karimireddy, S.P., Veit, A., Kim, S., Reddi, S.J., Kumar, S. and Sra, S., 2019. Why are adaptive methods good for attention models?. arXiv preprint arXiv:1912.03194.](https://arxiv.org/abs/1912.03194)


<a id="3">[3]</a> 
[Ioffe, S. and Szegedy, C., 2015, June. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning (pp. 448-456). PMLR.](http://proceedings.mlr.press/v37/ioffe15.html)







