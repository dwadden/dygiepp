Pull requests and contributions are more than welcome. Here are a list of things that would be great to have done. These tasks could be a good fit for, for instance, undergrads interesting in getting into NLP.

- **Enable multi-GPU training and prediction**. There's a [tutorial](https://medium.com/ai2-blog/tutorial-how-to-train-with-multiple-gpus-in-allennlp-c4d7c17eb6d6) on how to do this.
- **Re-factor and comment the modeling code**. Basically all of the modeling is accomplished by enumerating spans, and then running them through unary or binary scoring functions. Because this was written as research code, a lot of functionality is duplicated. Re-factoring could make the code much easier to extend. If interested, email me and I'll provide more info.
