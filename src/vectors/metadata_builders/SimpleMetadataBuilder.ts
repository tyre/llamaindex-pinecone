import { BaseNode } from 'llamaindex';
import { PineconeMetadata } from 'pinecone_api';
import { PineconeMetadataBuilder } from './types';
import { validateMetadata } from './utils';

export type SimpleMetadataBuilderOptions = {
  excludedMetadataKeys?: string[];
};

export class SimpleMetadataBuilder implements PineconeMetadataBuilder {
  excludedMetadataKeys: string[] = [];

  constructor(options: SimpleMetadataBuilderOptions = {}) {
    Object.assign(this, options);
  }

  buildMetadata(node: BaseNode): PineconeMetadata {
    return Object.entries(node.metadata).reduce((metadata, [key, value]): PineconeMetadata => {
      if (this.excludedMetadataKeys.includes(key)) {
        return metadata;
      }
      validateMetadata(key, value);
      metadata[key] = node.metadata[key];
      return metadata;
    }, { nodeId: node.nodeId } as PineconeMetadata);
  }
}
