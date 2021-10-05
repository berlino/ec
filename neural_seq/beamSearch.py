import torch

from neural_seq.decoderUtils import *
import heapq

MAX_PROGRAM_LENGTH = 20

def pop_node_to_expand(nodes, epsilon):

    if random.random() < epsilon:
        i = random.randint(0,len(nodes)-1)
        node = nodes[i]
        nodes[i] = nodes[0]
        heapq.heappop(nodes)
    else:
        node = heapq.heappop(nodes)
    return node


def randomized_beam_search_decode(decoder, encoderOutput, rnn_decode, restrict_types, beam_width, epsilon, device):

    if not restrict_types:
        raise Exception("Not Implemented")

    num_end_nodes = beam_width

    # list of final programs
    endNodes = []

    if rnn_decode:
        # we assume num_layers=1 and batch_size=1
        assert decoder.num_layers == 1 and encoderOutput.size(0) == 1
        # num_layers x batch_size x embed_dim
        init_hidden = decoder.bridge(encoderOutput)
        init_hidden = init_hidden.reshape(1,1,-1) if rnn_decode else None
    else:
        init_hidden = None

    # starting node
    nodes = [PartialProgram(decoder.primitiveToIdx, decoder.request.returns(), device, hidden=init_hidden)]

    while True:
        newNodes = []
        for k in range(beam_width):

            # if beam is empty and there are no newly expanded nodes stop
            if len(nodes) == 0:
                if len(newNodes) == 0:
                    return endNodes
                else:
                    break
            node = pop_node_to_expand(nodes, epsilon)
    
            if rnn_decode:
                parenTokenIdx = torch.tensor([node.parentTokenStack.pop()], device=device)
                probs, hidden, nextTokenType, lambdaVars, node = decoder.forward_rnn(encoderOutput, node, parenTokenIdx,
                last_hidden=node.hidden, device=device, restrictTypes=True)

            else:
                # batch_size (1) x embed_dim
                parenTokenIdx = torch.tensor([node.parentTokenStack.pop()], device=device)
                probs, nextTokenType, lambdaVars, node = decoder.forward(encoderOutput, node, parenTokenIdx,
                device=device, restrictTypes=True)
                hidden = None

            # assumes batch_size of 1
            probs = probs.squeeze()
            indices = probs.nonzero()

            # expand node adding all possible next partial programs to queue (newNodes)
            for idx in indices:
                nextToken = decoder.idxToPrimitive[idx.item()]
                newNode = node.copy()
                nllScore = -torch.log(probs[idx])
                newNode.processNextToken(nextToken, nextTokenType, nllScore, lambdaVars, decoder.primitiveToIdx, hidden, device)

                if len(newNode.nextTokenTypeStack) == 0:
                    endNodes.append((newNode.totalScore, newNode))
                    # print("{} total nodes, Selected node has {} tokens, {} endNodes found".format(len(nodes), len(newNode.programTokenSeq), len(endNodes)))
                    # print("Decoded syntactically valid program: {}".format(newNode.programStringsSeq))
                    if len(endNodes) >= num_end_nodes:
                        return endNodes
                
                # using the type system we can be intelligent about ruling out programs that don't satisfy
                elif len(newNode.programTokenSeq) + len(newNode.nextTokenTypeStack) > MAX_PROGRAM_LENGTH:
                    continue
                else:
                    heapq.heappush(newNodes, newNode)
      
        nodes = newNodes
        # print("{} nodes in beam".format(len(nodes)))

    return endNodes


