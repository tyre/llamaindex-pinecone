export interface SparseValues {
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

export interface SparseValuesBuilder {
  embeddings: Array<number>;
  build(): SparseValues;
}

export type EmbeddingFrequencies = Record<string, number>;

export type SparseValuesBuilderClass = { new(embeddings: number[]): SparseValuesBuilder };