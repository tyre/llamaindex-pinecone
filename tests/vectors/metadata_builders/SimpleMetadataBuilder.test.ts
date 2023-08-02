import { SimpleMetadataBuilder } from "../../../src";
import { Document } from 'llamaindex';
import { PineconeMetadata } from 'pinecone_api';

describe("SimpleMetadataBuilder", () => {
  let node: Document;
  let builder: SimpleMetadataBuilder;

  beforeEach(() => {
    node = new Document({
      id_: "nodeId",
      metadata: {
        metadataKey: "metadataValue",
        metadataKey2: "metadataValue2"
      }
    });
    builder = new SimpleMetadataBuilder();
  });

  describe("buildMetadata", () => {
    it("should return a PineconeMetadata object with the nodeId and metadata", () => {
      const metadata: PineconeMetadata = builder.buildMetadata(node);
      expect(metadata).toEqual({
        nodeId: node.nodeId,
        ...node.metadata
      });
    });

    it("should throw an error if the metadata value is an object", () => {
      node.metadata = {
        metadataKey: { metadataValue: "metadataValue" }
      };
      expect(() => builder.buildMetadata(node)).toThrowError("Metadata value for metadataKey cannot be an object");
    });

    it("should allow the metadata value to be an array, but only with numbers, strings, and booleans", () => {
      node.metadata = {
        metadataKey: [1, "2", true]
      };
      expect(() => builder.buildMetadata(node)).not.toThrowError();
    });

    it("should throw an error if an array value is an object", () => {
      node.metadata = {
        metadataKey: [1, { metadataValue: "metadataValue" }, true]
      };
      expect(() => builder.buildMetadata(node)).toThrowError("Metadata value for member of metadataKey cannot be an object");
    });

    it("should not include a key if it is in the excludedMetadataKeys array", () => {
      builder = new SimpleMetadataBuilder({ excludedMetadataKeys: ["metadataKey"] });
      const metadata: PineconeMetadata = builder.buildMetadata(node);
      expect(metadata).toEqual({
        nodeId: node.nodeId,
        metadataKey2: "metadataValue2"
      });
    });
  });
});

