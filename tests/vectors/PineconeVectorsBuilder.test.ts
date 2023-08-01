import { NaiveSparseValuesBuilder, PineconeVectorsBuilder } from "../../src";
import { Document } from "llamaindex";

describe('PineconeVectorsBuilder', (): void => {
  const tongueTwister = new Document({
    text: "Peter Piper picked a peck of pickled peppers. A peck of pickled peppers Peter Piper picked. If Peter Piper picked a peck of pickled peppers, Where's the peck of pickled peppers Peter Piper picked?",
    id_: "peter-piper",
    metadata: {
      type: "toungue-twister",
      language: "english"
    }
  });
  const embeddings = [
    1, 5310, 7362, 546, 18691, 263, 1236, 384, 310, 5839, 839, 1236,
    22437, 29889, 319, 1236, 384, 310, 5839, 839, 1236, 22437, 5310, 7362,
    546, 18691, 29889, 960, 5310, 7362, 546, 18691, 263, 1236, 384, 310,
    5839, 839, 1236, 22437, 29892, 6804, 29915, 29879, 278, 1236, 384, 310,
    5839, 839, 1236, 22437, 5310, 7362, 546, 18691, 29973
  ]; // length 57

  it('should generate vector when dimension is same length as embeddings', (): void => {
    const builder = new PineconeVectorsBuilder(tongueTwister, embeddings, { dimension: embeddings.length });
    const vectors = builder.buildVectors();
    expect(vectors.length).toEqual(1);
    const vector = vectors[0];
    expect(vector.values.length).toEqual(embeddings.length);
    expect(vector.values).toEqual(embeddings);
    expect(vector.id).toEqual("peter-piper");
    expect(vector.metadata).toEqual({
      nodeId: "peter-piper",
      type: "toungue-twister",
      language: "english"
    });
  });

  describe('when strictDimensionCheck is false', (): void => {
    it('should generate multiple vectors when dimension is less than length of embeddings', (): void => {
      const builder = new PineconeVectorsBuilder(tongueTwister, embeddings, { dimension: 20, splitEmbeddingsByDimension: true });
      const vectors = builder.buildVectors();
      expect(vectors.length).toEqual(3);
      expect(vectors[0].id).toEqual("peter-piper-0");
      expect(vectors[0].values).toEqual(embeddings.slice(0, 20));
      expect(vectors[0].metadata).toEqual({
        nodeId: "peter-piper",
        type: "toungue-twister",
        language: "english"
      });

      expect(vectors[1].id).toEqual("peter-piper-1");
      expect(vectors[1].values).toEqual(embeddings.slice(20, 40));
      expect(vectors[1].metadata).toEqual({
        nodeId: "peter-piper",
        type: "toungue-twister",
        language: "english"
      });


      expect(vectors[2].id).toEqual("peter-piper-2");
      // We expect that final vector to include the other 17 embeddings then be padded with 3 zeros.
      expect(vectors[2].values).toEqual([
        29892, 6804, 29915, 29879,
        278, 1236, 384, 310,
        5839, 839, 1236, 22437,
        5310, 7362, 546, 18691,
        29973, 0, 0, 0
      ]);

      expect(vectors[2].metadata).toEqual({
        nodeId: "peter-piper",
        type: "toungue-twister",
        language: "english"
      });
    });

    it('should generate sparse values if requested', (): void => {
      const builder = new PineconeVectorsBuilder(
        tongueTwister,
        embeddings,
        { dimension: embeddings.length, includeSparseValues: true, splitEmbeddingsByDimension: true }
      );
      const vectors = builder.buildVectors();
      expect(vectors.length).toEqual(1);
      const sparseValues = vectors[0].sparseValues!;
      expect(sparseValues.indices.length).toEqual(new Set(sparseValues.indices).size);
      expect(sparseValues.indices).toEqual([
        1, 263, 278, 310, 319, 384, 546, 839,
        960, 1236, 5310, 5839, 6804, 7362, 18691, 22437,
        29879, 29889, 29892, 29915, 29973
      ])
      expect(sparseValues.values).toEqual([
        1, 2, 1, 4, 1, 4, 4, 4,
        1, 8, 4, 4, 1, 4, 4, 4,
        1, 2, 1, 1, 1
      ]);
    });

    it('should split sparse values when there are multiple vectors', (): void => {
      const builder = new PineconeVectorsBuilder(
        tongueTwister,
        embeddings,
        { dimension: 20, includeSparseValues: true, splitEmbeddingsByDimension: true }
      );
      const vectors = builder.buildVectors();
      expect(vectors.length).toEqual(3);
      expect(vectors[0].id).toEqual("peter-piper-0");
      expect(vectors[0].values).toEqual(embeddings.slice(0, 20));
      let sparseValueBuilder = new NaiveSparseValuesBuilder(vectors[0].values);
      expect(vectors[0].sparseValues).toEqual(sparseValueBuilder.build())

      expect(vectors[1].id).toEqual("peter-piper-1");
      expect(vectors[1].values).toEqual(embeddings.slice(20, 40));
      sparseValueBuilder = new NaiveSparseValuesBuilder(vectors[1].values);
      expect(vectors[1].sparseValues).toEqual(sparseValueBuilder.build())


      expect(vectors[2].id).toEqual("peter-piper-2");
      // We expect that final vector to include the other 17 embeddings then be padded with 3 zeros.
      expect(vectors[2].values).toEqual([
        29892, 6804, 29915, 29879,
        278, 1236, 384, 310,
        5839, 839, 1236, 22437,
        5310, 7362, 546, 18691,
        29973, 0, 0, 0
      ]);

      // Sparse values should not be calculated with any padding.
      sparseValueBuilder = new NaiveSparseValuesBuilder([
        29892, 6804, 29915, 29879,
        278, 1236, 384, 310,
        5839, 839, 1236, 22437,
        5310, 7362, 546, 18691,
        29973]);
      expect(vectors[2].sparseValues).toEqual(sparseValueBuilder.build())
    });
  });
})
