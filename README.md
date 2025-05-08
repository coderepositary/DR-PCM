# DR-PCM
Dual-space Relation-aware Entity Representation Learning for Personalized Compatibility Modeling

Existing methods encounter three limitations: (1) The deficiency in homogeneous relation modeling. (2) Limited ability to perceive dynamic relations between entities (i.e., users and items). And (3) indiscriminative fusion of the neighbor representation for information propagation. These limitations impede the dynamic perception of sampling strategies during entity iterations and the comprehensive aggregation of information among entities. To surmount these problems, we propose DR-PCM framework, which can enhance the representations by jointly modeling fashion entities in homogeneous and heterogeneous spaces and improve personalized recommendation performance with multi-modalities. Furthermore, we design a relation-aware dynamic neighbor sampling strategy, which perceives and learns the intricate interaction relations among entities and boosts the efficiency of neighbor node sampling. Subsequently, an adaptive feature fusion is performed from the dual space to precisely capture the varying confidences of the current representation and the neighbor representation for information propagation toward different target nodes.

本文设计的模型图如下:

![image](https://github.com/user-attachments/assets/f8c1b0c8-d53e-4c91-999e-21a0d9317384)
