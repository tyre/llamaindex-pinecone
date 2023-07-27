import { SparseValues, NaiveSparseValueBuilder } from '../../../../embeddings';
import assert from 'assert';

describe('NaiveSparseValueBuilder', () => {
  describe('build', () => {
    it('return the frequencies and values of the included embedder', () => {
      const embeddings = [1, 2, 3, 2, 3, 1, 5, 3, 1];
      const builder = new NaiveSparseValueBuilder(embeddings);
      const sparseValues = builder.build();
      assert.deepStrictEqual(sparseValues, { indicies: [3, 2, 3, 1], values: [1, 2, 3, 5] });
    });
  })
})