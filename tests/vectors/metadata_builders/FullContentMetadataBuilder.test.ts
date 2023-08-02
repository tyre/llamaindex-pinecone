import { Document } from "llamaindex";
import { PineconeMetadata } from "pinecone_api";
import { FullContentMetadataBuilder } from "../../../src/vectors/metadata_builders";


describe("FullContentMetadataBuilder", () => {
  it("should return a PineconeMetadata object with the nodeId, nodeType, and nodeContent", () => {
    const node = new Document({
      id_: "nodeId",
      metadata: {
        metadataKey: "metadataValue",
        metadataKey2: "metadataValue2"
      }
    });
    const builder = new FullContentMetadataBuilder();
    const metadata: PineconeMetadata = builder.buildMetadata(node);
    expect(metadata).toEqual({
      nodeId: node.nodeId,
      nodeType: node.getType(),
      nodeContent: JSON.stringify(node)
    });
  });
});