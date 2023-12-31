import { EmbeddingFrequencies, SparseValues, SparseValuesBuilder } from "./types";
/**
 * Builds sparse values using a naive approach.
 * 
 * Essentially:
 * 1. Build a map of embedding values to the frequency of that value.
 * 2. Build the sparse values object from the map.
 * 
 * Other methods, like BM25 or SPLADE, may be more effective.
 * 
 * @param {Array<number>} embeddingValues
 * @returns {SparseValues}
 * 
 * @example
 * ```ts
 * const embeddingValues = [1, 2, 3, 2, 3, 1, 5, 3, 1];
 * const sparseValues = buildSparseValues(embeddingValues);
 * // sparseValues = { indices: [1, 2, 3, 5], values: [3, 2, 3, 1] }
 * ```
*/
export class NaiveSparseValuesBuilder implements SparseValuesBuilder {
  embeddings: Array<number>;

  constructor(embeddings: Array<number>) {
    this.embeddings = embeddings;
  }

  /**
   * Build a sparse values object from an array of embedding values.
   * 
   * @returns {SparseValues}
   * 
   * @example
   * ```ts
   * const embeddingValues = [1, 2, 3, 2, 3, 1, 5, 3, 1];
   * const sparseValuesBuilder = new NaiveSparseValuesBuilder(embeddingValues);
   * const sparseValues = sparseValuesBuilder.build();
   * // sparseValues = { indices: [1, 2, 3, 5], values: [3, 2, 3, 1] }
   * ```
  */
  build(): SparseValues {
    const embeddingFrequencies = this.buildEmbeddingFrequencies();
    const sparseValues: SparseValues = { indices: [], values: [] };

    for (const token in embeddingFrequencies) {
      sparseValues.indices.push(parseInt(token));
      sparseValues.values.push(embeddingFrequencies[token]);
    }
    return sparseValues;
  }


  /**
     * Build a map of embedding values to their frequencies.
     * 
     * @param {Array<number>} embeddingValues
     * @returns {EmbeddingFrequencies}
     * 
  */
  private buildEmbeddingFrequencies(): EmbeddingFrequencies {
    return this.embeddings.reduce((acc, embedding) => {
      const embeddingKey = embedding.toString();
      if (!acc[embeddingKey]) {
        acc[embeddingKey] = 1;
      } else {
        acc[embeddingKey] += 1;
      }
      return acc;
    }, {} as EmbeddingFrequencies);
  }
}