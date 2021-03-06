# Second Project

 * In this project, I have implemented an **`Air Pollution Forcasting`** task utilizing **`GRU`**, **`LSTM`**, and **`RNN`** utilizing **`Keras`**.

* The project focused mainly on the following topics:

    -   Developing a **Recurrent Nueral Net** model for a **`regression`** task: daily, weekly, and monthly **Ari Pollution Forcasting**. 

    -   Investigaton the effect of various **Recurrent Neural Net** models: GRU, LSTM, and RNN on the final performance.

    - **`Feature Selection`** based on the correlation of features with **Ari Pollution**.

    - **`Model Enseble`** prediction utilzing different recurrent structures: GRU, LSTM, and RNN.

    - **`Missing Data Imputation`** and its effects on the final performance.

* **Notble Notes**: 
    -   On the ground [[1]](#1), [[2]](#2), we know that **LSTM** is nearly optimal, which is compatible with my simulations' results.
    -   I applied a linear layer on the trained GRU, RNN, and LSTM models in orde to fuse the models in **Model Ensemble** part.

* The codes are provided in [Codes](https://github.com/ARokni/Neural-Network-/tree/main/Project%202/Codes) folder, and the data available in [Dataset](https://github.com/ARokni/Neural-Network-/tree/main/Project%202/Dataset) folder.


## References
<a id="1">[1]</a> 
[Jozefowicz, R., Zaremba, W. and Sutskever, I., 2015, June. An empirical exploration of recurrent network architectures. In International conference on machine learning (pp. 2342-2350). PMLR.](http://proceedings.mlr.press/v37/jozefowicz15.html)

<a id="2">[2]</a> 
[Greff, K., Srivastava, R.K., Koutník, J., Steunebrink, B.R. and Schmidhuber, J., 2016. LSTM: A search space odyssey. IEEE transactions on neural networks and learning systems, 28(10), pp.2222-2232.](https://ieeexplore.ieee.org/abstract/document/7508408/)

