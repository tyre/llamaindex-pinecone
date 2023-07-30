import { QueryRequest } from "@pinecone-database/pinecone";
import { SparseValues } from "vectors";

export enum PineconeMetadataFilterKey {
  EqualTo = "$eq", // - Equal to (number, string, boolean)
  NotEqualTo = "$ne", // - Not equal to (number, string, boolean)
  GreaterThan = "$gt", // - Greater than (number)
  GreaterThanOrEqualTo = "$gte", // - Greater than or equal to (number)
  LessThan = "$lt", // - Less than (number)
  LessThanOrEqualTo = "$lte", // - Less than or equal to (number)
  In = "$in", // - In array (string or number)
  NotIn = "$nin" // - Not in array (string or number)
}

export type PineconeMetadataFilterValue = string | number | boolean | Array<string | number>;

export type PineconeMetadataFilter = {
  [key in PineconeMetadataFilterKey]: PineconeMetadataFilterValue;
}

export type PineconeQueryOptions = {
  namespace?: string;
  topK: number;
  includeValues?: boolean;
  includeMetadata?: boolean;
  vector?: number[];
  sparseVector?: SparseValues;
  filter?: PineconeMetadataFilter;
  id?: string;
}

export type PineconeQueryMatch = {
  id: string;
  score: number;
  values?: number[];
  sparseValues?: SparseValues;
  metadata?: Record<string, string | number | boolean>;
}

export type PineconeQueryResponse = {
  matches: Array<PineconeQueryMatch>;
  namespace: string;
}

export class PineconeQueryBuilder {
  namespace?: string;
  topK: number;
  includeValues: boolean = true;
  includeMetadata: boolean = true;
  vector?: number[];
  sparseVector?: SparseValues;
  filter?: PineconeMetadataFilter;
  id?: string;
  
  constructor(options: PineconeQueryOptions) {
    if (!options.id || !options.vector) {
      throw new Error('One of id and vector are required.');
    } else if (options.id && options.vector) {
      throw new Error('Only one of id and vector are allowed.');
    }
    this.topK = options.topK;
    Object.assign(this, options);
  }

  toQueryRequest(): QueryRequest {
    const queryRequest: Partial<QueryRequest> = {
      topK: this.topK,
      includeValues: this.includeValues,
      includeMetadata: this.includeMetadata,
    }

    // only add in keys that have values
    const maybePresentKeys: Array<keyof QueryRequest> = ["namespace", "vector", "filter", "id"];
    for (const key of maybePresentKeys) {
      if (this[key]) {
        const propertyKey = key as keyof PineconeQueryBuilder;
        queryRequest[key] = this[propertyKey];
      }
    }
    return queryRequest as QueryRequest;
  }
}