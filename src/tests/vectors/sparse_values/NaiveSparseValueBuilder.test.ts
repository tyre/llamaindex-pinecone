import { NaiveSparseValuesBuilder } from "../../../";
import assert from 'assert';

describe('NaiveSparseValuesBuilder', () => {
  describe('build', () => {
    it('return the frequencies and values of the included embedder', () => {
      const embeddings = [1, 2, 3, 2, 3, 1, 5, 3, 1];
      const builder = new NaiveSparseValuesBuilder(embeddings);
      const sparseValues = builder.build();
      assert.deepStrictEqual(sparseValues, { indices: [1, 2, 3, 5], values: [3, 2, 3, 1] });
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
      assert(sparseValues.indices.length === new Set(sparseValues.indices).size, "Indices are not unique!");
    });
  })
})