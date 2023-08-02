import {
  BaseNode,
  ObjectType as LlamaNodeType,
  Document as LlamaDocument,
  IndexNode as LlamaIndexNode,
  TextNode as LlamaTextNode
} from "llamaindex";
import { PineconeMetadata } from "pinecone_api";


export type NodeHydratorClass = {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  new: (args?: any) => NodeHydrator;
}

export type NodeHydrator = {
  hydrate: (vectorMetadata: PineconeMetadata) => BaseNode;
}

// Hydrates a vector's metadata into a full node.
// Expects that the node's content is a JSON string inside
// of `nodeContent` in the vector's metadata.
export class FullContentNodeHydrator implements NodeHydrator {
  hydrate(vectorMetadata: PineconeMetadata): BaseNode {
    const nodeContent = vectorMetadata.nodeContent as string
    if (!nodeContent) {
      throw new Error(`Vector for node ${vectorMetadata.nodeId} has no nodeContent key in its metadata.`);
    }
    const nodeAsJson = JSON.parse(nodeContent);
    switch (vectorMetadata.nodeType as LlamaNodeType) {
      case LlamaNodeType.DOCUMENT:
        return new LlamaDocument(nodeAsJson);
      case LlamaNodeType.INDEX:
        return new LlamaIndexNode(nodeAsJson);
      case LlamaNodeType.TEXT:
        return new LlamaTextNode(nodeAsJson);
      default:
        throw new Error(`Unknown node type ${vectorMetadata.nodeType}`);
    }
  }
}