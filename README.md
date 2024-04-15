# Statistical-Language-Models-for-Scene-Understanding

A significant problem in robotics is understanding sceneries, which calls for robots to
be able to semantically comprehend spaces and their contents in addition to being able to
navigate and localize in a variety of situations. For example, if a robot is told "go fetch a spoon,"
it should be able to accomplish the task because it is naturally familiar with the common items
present in a kitchen. An underutilized strategy to deal with this is to employ language models,
which, when trained on substantial amounts of text data, collect semantic information. As an
illustration, a language model might come to correlate "Bathrooms contain..." with "toilets"
rather than "stoves," exhibiting a degree of common sense that is essential for comprehending
the setting. Furthermore, language's ability to convey commonsense questions, even those
incorporating novel concepts is crucial for spatial perception since deployed robots can come
across unexpected items and need to be able to conclude unknown object kinds.
This project aims to develop an image classification model, based on objects detected
in the scene. We took motivation from the paper ‘Extracting Zero-shot Common Sense from
Large Language Models for Robot 3D Scene Understanding’ [21], which utilizes a large
language model and zero-shot learning approach to generalize to arbitrary room and object
labels, including those that it has not seen during training. Similarly, this project aims to utilize
Language Models employing the “Bag of Words” technique to classify rooms based on the
objects detected within the scenes captured in images. Though we took a similar approach as
[21] our approach is less complex and easy to deploy.
The significance of this approach is that it provides the model with a semantic
understanding of an environment and the entities within it. To explain this, we can revisit the
above example “go fetch a spoon,” the model should understand that a spoon is likely to be in
the kitchen rather than the bedroom. Another significance of this approach might have is to
reduce the training time or complexity of Computer Vision models by limiting the training data
to only lower-level classes i.e., specific objects. In the popular ImageNet challenges new and
efficient models are being introduced every year for detection or classification tasks. These
models almost always are trained on data that have labels on multiple hierarchical levels. For
example, a class car could have sub-classes called Fiet, BMW, Audi, etc. This always makes
the model more complex. In our approach, we hope to create a new method for classification
models that use the lower-level classes to classify classes on higher levels.
