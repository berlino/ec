import torch

from larc.decoderUtils import *
import heapq

MAX_PROGRAM_LENGTH = 30

def randomized_beam_search_decode(decoder, encoderOutput, beam_width, epsilon, num_end_nodes):

    # list of final programs
    endNodes = []

    # starting node
    nodes = [PartialProgram(decoder.primitiveToIdx, decoder.request.returns(), decoder.device)]

    while True:

        for k in range(beam_width):
            # true with probability epsilon and false with probability (1-epsilon)
            if random.random() < epsilon:
                 i = random.randint(0,len(nodes)-1)
                 node = nodes[i]
                 nodes[i] = nodes[0]
                 heapq.heappop(nodes)
            else:
                 node = heapq.heappop(nodes)

            # print("{} total nodes, Selected node has {} tokens, {} endNodes found".format(len(nodes), len(node.programTokenSeq), len(endNodes)))
            # print("Selected node: {}".format(node.programStringsSeq))
            attnOutputWeights, nextTokenType, lambdaVars, node = decoder.forward(encoderOutput, node)
            # print('pp', node.programStringsSeq)
            # print("nextTokenType", nextTokenType)
            # print(attnOutputWeights)
            attnOutputWeights = attnOutputWeights[0, 0, :]

            weights, indices = attnOutputWeights[attnOutputWeights > 0], attnOutputWeights.nonzero()
            # print(indices)
            # print([decoder.idxToPrimitive[idx.item()] for idx in indices])

            # add to queue
            for idx in indices:
                nextToken = decoder.idxToPrimitive[idx.item()]
                newNode = node.copy()
                newNode.processNextToken(nextToken, nextTokenType, torch.log(attnOutputWeights[idx]), lambdaVars, decoder.primitiveToIdx, decoder.device)

                if len(newNode.nextTokenTypeStack) == 0:
                    endNodes.append((newNode.totalScore, newNode))
                    # print("{} total nodes, Selected node has {} tokens, {} endNodes found".format(len(nodes), len(newNode.programTokenSeq), len(endNodes)))
                    # print("Selected node: {}".format(newNode.programStringsSeq))
                    if len(endNodes) >= num_end_nodes:
                        return endNodes

                elif len(newNode.programTokenSeq) > MAX_PROGRAM_LENGTH:
                    continue
                else:
                    heapq.heappush(nodes, newNode)
    return endNodes


