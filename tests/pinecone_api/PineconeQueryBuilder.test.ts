import { PineconeQueryBuilder } from "../../src/pinecone_api/PineconeQueryBuilder";

describe('PineconeQueryBuilder', (): void => {
  it('should throw an error if neither id or vector are provided', (): void => {
    expect(() => {
      new PineconeQueryBuilder({ topK: 1 });
    }).toThrow();
  });

  it('should throw an error if both id and vector are provided', (): void => {
    expect(() => {
      new PineconeQueryBuilder({ topK: 1, id: "test", vector: [1, 2, 3] });
    }).toThrow();
  });

  it('converts to a query request', (): void => {
    const builder = new PineconeQueryBuilder({ topK: 1, id: "test" });
    const queryRequest = builder.toQueryRequest();
    expect(queryRequest).toEqual({
      topK: 1,
      id: "test",
      includeValues: true,
      includeMetadata: true
    });
  });

  it('converts to a query request with a namespace', (): void => {
    const builder = new PineconeQueryBuilder({ topK: 1, id: "test", namespace: "test-namespace" });
    const queryRequest = builder.toQueryRequest();
    expect(queryRequest).toEqual({
      topK: 1,
      id: "test",
      namespace: "test-namespace",
      includeValues: true,
      includeMetadata: true
    });
  });
});