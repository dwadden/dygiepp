import numpy
import math

from allennlp.interpret.saliency_interpreters.integrated_gradient import IntegratedGradient
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter
from allennlp.common.util import JsonDict, sanitize


@SaliencyInterpreter.register("dygie")
class DygieInterpreter(IntegratedGradient):
    """
    Does the same thing as the integrated gradients interpreter, except that it takes labeled
    instances as input instead of json. This makes it interoperate better with the predictor.
    """
    def saliency_interpret_from_labeled_instances(self, labeled_instances):
        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            # Run integrated gradients
            grads = self._integrate_gradients(instance)

            # Normalize results
            for key, grad in grads.items():
                # The [0] here is undo-ing the batching that happens in get_gradients.
                embedding_grad = numpy.sum(grad[0], axis=1)
                norm = numpy.linalg.norm(embedding_grad, ord=1)
                normalized_grad = [math.fabs(e) / norm for e in embedding_grad]
                grads[key] = normalized_grad

            instances_with_grads['instance_' + str(idx + 1)] = grads

        return sanitize(instances_with_grads)
