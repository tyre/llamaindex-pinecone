import { BaseNode } from 'llamaindex';
import { PineconeMetadata } from 'pinecone_api';
import { PineconeMetadataBuilder } from './types';

export type SimpleMetadataBuilderOptions = {
  excluedMetadataKeys?: string[];
};

export class SimpleMetadataBuilder implements PineconeMetadataBuilder {
  excluedMetadataKeys: string[] = [];

  constructor(options: SimpleMetadataBuilderOptions = {}) {
    Object.assign(this, options);
  }

  buildMetadata(node: BaseNode): PineconeMetadata {

    return Object.entries(node.metadata).reduce((metadata, [key, value]): PineconeMetadata => {
      if (this.excluedMetadataKeys.includes(key)) {
        return metadata;
      } else if (typeof key !== "string") {
        throw new Error(`Metadata key ${key} must be a string`);
      } else if (isAnObject(value)) {
        throw new Error(`Metadata value for ${key} cannot be an object`);
      } else if (isAnArray(value)) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        value.forEach((arrayValue: any) => {
          if (isAnObject(arrayValue)) {
            throw new Error(`Metadata value for member of ${key} cannot be an object`);
          }
        });
      }

      metadata[key] = node.metadata[key];
      return metadata;
    }, { nodeId: node.nodeId } as PineconeMetadata);
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function isAnArray(value: any): boolean {
  return Array.isArray(value);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function isAnObject(value: any): boolean {
  return typeof value === "object" && !isAnArray(value);
}