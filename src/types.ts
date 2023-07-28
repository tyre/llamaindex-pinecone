interface SparseValues {
  /**
   * The indices of the sparse data.
   * @type {Array<number>}
   * @memberof SparseValues
   */
  indices: Array<number>;
  /**
   * The corresponding values of the sparse data
   * Values array must have the same length as the indices array.
   * @type {Array<number>}
   * @memberof SparseValues
   */
  values: Array<number>;
}

interface SparseValueBuilder {
  embeddings: Array<number>;
  build(): SparseValues;
}

type EmbeddingFrequencies = Record<string, number>;
