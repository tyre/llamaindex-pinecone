import { NaiveSparseValuesBuilder } from "../../../src/vectors/sparse_values";

describe('NaiveSparseValuesBuilder', () => {
  describe('build', () => {
    it('return the frequencies and values of the included embedder', () => {
      const embeddings = [1, 2, 3, 2, 3, 1, 5, 3, 1];
      const builder = new NaiveSparseValuesBuilder(embeddings);
      const sparseValues = builder.build();
      expect(sparseValues).toStrictEqual({ indices: [1, 2, 3, 5], values: [3, 2, 3, 1] });
    });

    it('has unique indicies', () => {
      const embeddings = [
        1, 5310, 7362, 546, 18691, 263,
        1236, 384, 310, 5839, 839, 1236,
        22437, 29889, 319, 1236, 384, 310,
        5839, 839, 1236, 22437, 5310, 7362,
        546, 18691, 29889, 960, 5310, 7362,
        546, 18691, 263, 1236, 384, 310,
        5839, 839, 1236, 22437, 29892, 6804,
        29915, 29879, 278, 1236, 384, 310,
        5839, 839, 1236, 22437, 5310, 7362,
        546, 18691, 29973
      ]
      const builder = new NaiveSparseValuesBuilder(embeddings);
      const sparseValues = builder.build();

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
  })
})