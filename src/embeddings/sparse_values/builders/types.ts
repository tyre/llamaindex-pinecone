import { SparseValues } from "../types";

export interface SparseValueBuilder {
  embeddings: Array<number>;
  build(): SparseValues;
}