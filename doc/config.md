The configuration process for DyGIE relies on the `jsonnet`-based configuration system for [AllenNLP](https://guide.allennlp.org/using-config-files). For more information on the AllenNLP configuration process in general, take a look at the AllenNLP [guide](https://guide.allennlp.org).

DyGIE adds one (regrettable but unavoidable) layer of complexity on top of this. It factors the configuration into:

- Components that are common to all DyGIE models. These are defined in [training_config/template.libsonnet].
- Components that are specific to single model trained on a particular dataset. These are contained in the `jsonnet` files in [training_config]. They use the jsonnet inheritance mechanism to extend the base class defined in `template.libsonnet`.  For more on jsonnet inheritance, see the [jsonnet tutorial](https://jsonnet.org/learning/tutorial.html)


Add a note about adding a predefined vocab. It's like
```
    vocabulary: {
      type: "from_files",
      directory: p.vocab_path // config changed TODO
    },

```

Also add instances_per_epoch
