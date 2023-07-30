import { SparseValues } from "index";

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

export type PineconeMetadataFilterValue = string | number | boolean | Array<string | number | PineconeMetadataFilter>;

export type PineconeMetadataFilter = {
  [key in PineconeMetadataFilterKey]: PineconeMetadataFilterValue;
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

export type PineconeUpsertOptions = {
  batchSize?: number;
  includeSparseValues?: boolean;
}

export type PineconeUpsertResults = {
  upsertedCount: number;
  failedCount: number;
  upsertedVectorIds: string[];
  failedVectorIds: string[];
}

export type PineconeEnv = {
  apiKey: string;
  environment: string;
}