import {
  ObjectType as LlamaNodeType,
  Document as LlamaDocument,
  IndexNode as LlamaIndexNode,
  TextNode as LlamaTextNode
} from "llamaindex";
import { PineconeMetadata } from "pinecone_api";
import { FullContentNodeHydrator } from '../../../src/vectors/hydrators';

// export type NodeHydratorClass = {
//   // eslint-disable-next-line @typescript-eslint/no-explicit-any
//   new: (args?: any) => NodeHydrator;
// }

// export type NodeHydrator = {
//   hydrate: (vectorMetadata: PineconeMetadata) => BaseNode;
// }

// // Hydrates a vector's metadata into a full node.
// // Expects that the node's content is a JSON string inside
// // of `nodeContent` in the vector's metadata.
// export class FullContentNodeHydrator implements NodeHydrator {
//   hydrate(vectorMetadata: PineconeMetadata): BaseNode {
//     const nodeContent = vectorMetadata.nodeContent as string
//     if (!nodeContent) {
//       throw new Error(`Vector for node ${vectorMetadata.nodeId} has no nodeContent key in its metadata.`);
//     }
//     const nodeAsJson = JSON.parse(nodeContent);

//     switch (vectorMetadata.nodeType as LlamaNodeType) {
//       case LlamaNodeType.DOCUMENT:
//         return new LlamaDocument(nodeAsJson);
//       case LlamaNodeType.INDEX:
//         return new LlamaIndexNode(nodeAsJson);
//       case LlamaNodeType.TEXT:
//         return new LlamaTextNode(nodeAsJson);
//       default:
//         throw new Error(`Unknown node type ${vectorMetadata.nodeType}`);
//     }
//   }
// }

describe('FullContentNodeHydrator', () => {
  const nodeHydrator = new FullContentNodeHydrator();

  describe('hydrate', () => {
    it('hydrates a document node', () => {
      const vectorMetadata: PineconeMetadata = {
        nodeId: 'nodeId',
        nodeType: LlamaNodeType.DOCUMENT,
        nodeContent: JSON.stringify({
          id_: 'nodeId',
          text: 'content'
        })
      };

      const node = nodeHydrator.hydrate(vectorMetadata);

      expect(node).toBeInstanceOf(LlamaDocument);
      expect(node.nodeId).toEqual(vectorMetadata.nodeId);
      expect((node as LlamaDocument).text).toEqual('content');
    });

    it('hydrates an index node', () => {
      const vectorMetadata: PineconeMetadata = {
        nodeId: 'nodeId',
        nodeType: LlamaNodeType.INDEX,
        nodeContent: JSON.stringify({
          id_: 'nodeId',
          indexId: 'index-id'
        })
      };
      console.log({ vectorMetadata });
      const node = nodeHydrator.hydrate(vectorMetadata);

      expect(node).toBeInstanceOf(LlamaIndexNode);
      expect(node.nodeId).toEqual(vectorMetadata.nodeId);
      console.dir({ node }, { depth: null });
      expect((node as LlamaIndexNode).indexId).toEqual('index-id');
    });

    it('hydrates a text node', () => {
      const vectorMetadata: PineconeMetadata = {
        nodeId: 'nodeId',
        nodeType: LlamaNodeType.TEXT,
        nodeContent: JSON.stringify({
          id_: 'nodeId',
          text: 'some text!'
        })
      };

      const node = nodeHydrator.hydrate(vectorMetadata);

      expect(node).toBeInstanceOf(LlamaTextNode);
      expect(node.nodeId).toEqual(vectorMetadata.nodeId);
      expect((node as LlamaTextNode).text).toEqual('some text!');
    });

    it('throws an error if nodeContent is not present', () => {
      const vectorMetadata: PineconeMetadata = {
        nodeId: 'nodeId',
        nodeType: LlamaNodeType.TEXT,
      };

      expect(() => nodeHydrator.hydrate(vectorMetadata)).toThrowError();
    });

    it('throws an error if nodeType is not present', () => {
      const vectorMetadata: PineconeMetadata = {
        nodeId: 'nodeId',
        nodeContent: JSON.stringify({
          content: 'content'
        })
      };

      expect(() => nodeHydrator.hydrate(vectorMetadata)).toThrowError();
    });

    it('throws an error if nodeType is not a valid node type', () => {
      const vectorMetadata: PineconeMetadata = {
        nodeId: 'nodeId',
        nodeType: 'invalid-node-type',
        nodeContent: JSON.stringify({
          content: 'content'
        })
      };

      expect(() => nodeHydrator.hydrate(vectorMetadata)).toThrowError();
    });
  });
});