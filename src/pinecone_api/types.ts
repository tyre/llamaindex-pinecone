import { SparseValues } from "index";
import { Vector as PineconeVector } from "@pinecone-database/pinecone";


export type PineconeMetadata = Record<string, string | number | boolean | Array<string>>;

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

type PineconeMetadataFilterEqualTo = Record<PineconeMetadataFilterKey.EqualTo, number | string | boolean>;
type PineconeMetadataFilterNotEqualTo = Record<PineconeMetadataFilterKey.NotEqualTo, number | string | boolean>;
type PineconeMetadataFilterGreaterThan = Record<PineconeMetadataFilterKey.GreaterThan, number>;
type PineconeMetadataFilterGreaterThanOrEqualTo = Record<PineconeMetadataFilterKey.GreaterThanOrEqualTo, number>;
type PineconeMetadataFilterLessThan = Record<PineconeMetadataFilterKey.LessThan, number>;
type PineconeMetadataFilterLessThanOrEqualTo = Record<PineconeMetadataFilterKey.LessThanOrEqualTo, number>;
type PineconeMetadataFilterIn = Record<PineconeMetadataFilterKey.In, Array<string | number>>;
type PineconeMetadataFilterNotIn = Record<PineconeMetadataFilterKey.NotIn, Array<string | number>>;

export type PineconeMetadataFilter = PineconeMetadataFilterEqualTo |
  PineconeMetadataFilterNotEqualTo |
  PineconeMetadataFilterGreaterThan |
  PineconeMetadataFilterGreaterThanOrEqualTo |
  PineconeMetadataFilterLessThan |
  PineconeMetadataFilterLessThanOrEqualTo |
  PineconeMetadataFilterIn |
  PineconeMetadataFilterNotIn;

// Filters are of the form {"metadataPropertyKey": {"metadataFilterType": "value"}} where the value
// is either a scalar or an array of scalars.
export type PineconeMetadataFilters = Record<string, PineconeMetadataFilter>;

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
  splitEmbeddingsByDimension?: boolean;
}

export type PineconeUpsertResults = {
  upsertedNodeCount: number;
  upsertedNodeIds: string[];
  upsertedVectorByNode: Record<string, PineconeVector[]>;
  upsertedVectorCount: number;
  failedNodeCount: number;
  failedNodeIds: string[];
  errors: string[];
}

export type PineconeEnv = {
  apiKey: string;
  environment: string;
}

export type NodePineconeVectorMap = {
  [nodeId: string]: Array<PineconeVector>;
}

export type PineconeUpsertVectorsRecord = {
  totalVectorCount: number;
  vectorsByNode: NodePineconeVectorMap;
}


