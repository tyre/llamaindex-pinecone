import { BaseNode } from 'llamaindex';
import { PineconeMetadata } from 'pinecone_api';
import { PineconeMetadataBuilder } from './types';

export class SimpleMetadataBuilder implements PineconeMetadataBuilder {
  buildMetadata(node: BaseNode): PineconeMetadata {
    return {
      nodeId: node.nodeId,
      ...node.metadata
    };
  }
}