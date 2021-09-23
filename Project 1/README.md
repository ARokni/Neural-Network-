# First Project
 
 * In this project, I have implemented a `Multi-Category Image Classification` task via `CNN` models on `German Traffic Sign Recognition Benchmark` dataset utilizing `Keras`.

 * The project focused mainly on the following topics:

    -   Developing a **CNN** model for the classification task

    -   Investigation the effect of various `Activation Functions`:`ReLU`, `tanh`, and `Sigmoid`,  on the final model's performance.

    - Investigation the effect of various `Optimizers`: `Gradient Descent` and `Adam`, on the final model's performance.
    
    - Investigation the effect of `Dropout` and `Batch Normalization` on the final model's performance.

    - Investigation the effect of `Data Augmentaton` on the final model's performance.

    - Providing `Loss Accuracy` for `Validation` and `Test` data and depicting `Confusion Matrix` for the profound assement of the ability of the model in prediction of each seperate class.

* Notble Notes:
    -  On the ground of [[1]](#1), I have expected **ReLU** activation outperform **Sigmoid** and **tanh** since ReLu does not saturate and is computationally efficient. Moreover,  ReLu converges much fsate than Sigmoid/tanh in practice(e.g. 6X). By simulation I figured out the results are consistent with the reference paper.

    -  Based on [[2]](#2) , we know that **Gradient Descent** optimizer outperfroms **Adaptive** methods in **Image Processing** tasks; however, I applied **Adam** since adaptive optimizers have a higher convergence rate.

    - In **Data Augmentation** I found that **rotation** results in poor performance of the final model. In fact, since some different **Traffic Signs** are symmetric, applying the rotation results in amiguity and misunderstaing in models. As a result I eliminated the roation transformation.



