import { BaseNode } from "llamaindex";
import { PineconeMetadata } from "pinecone_api";
import { PineconeMetadataBuilder } from "./types";

export class FullContentMetadataBuilder implements PineconeMetadataBuilder {
  buildMetadata(node: BaseNode): PineconeMetadata {
    const nodeContent = JSON.stringify(node);
    const metadata: PineconeMetadata = {
      nodeId: node.id_,
      nodeType: node.getType(),
      nodeContent,
    };
    return metadata;
  }
}

