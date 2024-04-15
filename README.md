# Statistical-Language-Models-for-Scene-Understanding

This project aims to develop an image classification model, based on objects detected in the scene. We took motivation from the paper ‘Extracting Zero-shot Common Sense from Large Language Models for Robot 3D Scene Understanding’ which utilizes a large language model and zero-shot learning approach to generalize to arbitrary room and object labels, including those that it has not seen during training. Similarly, this project aims to utilize Language Models employing the “Bag of Words” technique to classify rooms based on the objects detected within the scenes captured in images. 


The significance of this approach is that it provides the model with a semantic understanding of an environment and the entities within it. To explain this, we can revisit the above example “go fetch a spoon,” the model should understand that a spoon is likely to be in the kitchen rather than the bedroom. Another significance of this approach might have is to reduce the training time or complexity of Computer Vision models by limiting the training data to only lower-level classes i.e., specific objects. In the popular ImageNet challenges new and efficient models are being introduced every year for detection or classification tasks. These models almost always are trained on data that have labels on multiple hierarchical levels. For example, a class car could have sub-classes called Fiet, BMW, Audi, etc. This always makes the model more complex. In our approach, we hope to create a new method for classification models that use the lower-level classes to classify classes on higher levels.

Contributors:
Avinash Arutla, Laleth Indirani Nehrukumar, Sheela Hansda

arutla.a@northeastern.edu, indiraninehrukumar.l@northeastern.edu, hansda.s@northeastern.edu
