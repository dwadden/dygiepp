import numpy
import math
from overrides import overrides

from allennlp.interpret.saliency_interpreters.simple_gradient import SimpleGradient
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter
from allennlp.common.util import sanitize
from allennlp.nn import util


@SaliencyInterpreter.register("dygie")
class DyGIEInterpreter(SimpleGradient):
    def saliency_interpret_from_labeled_instances(self, labeled_instances):
        """
        Same thing as the base class, except working from labeled instances instead of json.
        """
        embeddings_list = []

        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            # Hook used for saving embeddings
            handle = self._register_forward_hook(embeddings_list)
            grads = self.predictor.get_gradients([instance])[0]
            handle.remove()

            # Gradients come back in the reverse order that they were sent into the network
            embeddings_list.reverse()
            for key, grad in grads.items():
                # Get number at the end of every gradient key (they look like grad_input_[int],
                # we're getting this [int] part and subtracting 1 for zero-based indexing).
                # This is then used as an index into the reversed input array to match up the
                # gradient and its respective embedding.
                input_idx = int(key[-1]) - 1
                # The [0] here is undo-ing the batching that happens in get_gradients.
                emb_grad = numpy.sum(grad[0] * embeddings_list[input_idx], axis=1)
                norm = numpy.linalg.norm(emb_grad, ord=1)
                normalized_grad = [math.fabs(e) / norm for e in emb_grad]
                grads[key] = normalized_grad

            instances_with_grads['instance_' + str(idx + 1)] = grads
        return sanitize(instances_with_grads)

    @overrides
    def _register_forward_hook(self, embeddings_list):
        """
        Same as in the original code, except we need to move onto CPU.
        """
        def forward_hook(module, inputs, output):  # pylint: disable=unused-argument
            embeddings_list.append(output.squeeze(0).clone().detach().cpu().numpy())

        embedding_layer = util.find_embedding_layer(self.predictor._model)
        handle = embedding_layer.register_forward_hook(forward_hook)

        return handle
