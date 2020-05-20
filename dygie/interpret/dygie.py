import numpy
import math
from overrides import overrides

from allennlp.interpret.saliency_interpreters.integrated_gradient import IntegratedGradient
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter
from allennlp.common.util import JsonDict, sanitize
from allennlp.nn import util


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

    @overrides
    def _register_forward_hook(self, alpha, embeddings_list):
        # NOTE(dw) copy-pasted from the `integrated_gradients.py` code.
        def forward_hook(module, inputs, output):  # pylint: disable=unused-argument
            # Save the input for later use. Only do so on first call.
            if alpha == 0:
                # NOTE(dwadden) Need to switch to CPU; the original code didn't deal with this.
                embeddings_list.append(output.squeeze(0).clone().detach().cpu().numpy())

            # Scale the embedding by alpha
            output.mul_(alpha)

        # Register the hook
        embedding_layer = util.find_embedding_layer(self.predictor._model)
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle
