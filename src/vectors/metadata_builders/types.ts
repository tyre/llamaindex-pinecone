import { BaseNode } from "llamaindex";
import { PineconeMetadata } from "pinecone_api";

export interface PineconeMetadataBuilderClass {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  new(args?: any): PineconeMetadataBuilder;
}

export interface PineconeMetadataBuilder {
  buildMetadata(node: BaseNode): PineconeMetadata;
}