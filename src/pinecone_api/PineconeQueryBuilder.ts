import { QueryRequest } from "@pinecone-database/pinecone";
import { SparseValues } from "vectors";
import { PineconeMetadataFilter } from "pinecone_api/types";

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

// Because we're storing properties directly on the PineconeQueryBuilder,
// we build a type that is the intersection of on the builder and the
// query request. When building the final representation,
// we can ensure that the only builder properties allowed to be added are those
// which also exist on a pinecone QueryRequest
type OptionalRequestProperty = Array<
  keyof PineconeQueryBuilder & keyof QueryRequest
>;

export class PineconeQueryBuilder {
  namespace?: string;
  topK: number;
  includeValues: boolean = true;
  includeMetadata: boolean = true;
  vector?: number[];
  sparseVector?: SparseValues;
  filter?: PineconeMetadataFilter;
  id?: string;

  private OPTIONAL_QUERY_REQUEST_PROPERTIES: OptionalRequestProperty =
    ["namespace", "vector", "filter", "id"];

  constructor(options: PineconeQueryOptions) {
    if (!options.id && !options.vector) {
      throw new Error('One of `id` or `vector` are required.');
    } else if (options.id && options.vector) {
      throw new Error('Only one of `id` and `vector` are allowed.');
    }
    this.topK = options.topK;
    Object.assign(this, options);
  }

  toQueryRequest(): QueryRequest {
    let newQueryRequest: Partial<QueryRequest> = {
      topK: this.topK,
      includeValues: this.includeValues,
      includeMetadata: this.includeMetadata,
    }
    newQueryRequest = this.addOptionalQueryRequestValues(newQueryRequest);
    return newQueryRequest as QueryRequest;
  }

  addOptionalQueryRequestValues(queryRequest: Partial<QueryRequest>): Partial<QueryRequest> {
    // only add in keys that have values, and only allow keys that
    // are both properties of this object and on the pinecone QueryRequest
    return this.OPTIONAL_QUERY_REQUEST_PROPERTIES.reduce((_prev, key): Partial<QueryRequest> => {
      if (this[key]) {
        const requestPropertyKey = key as keyof QueryRequest;

        // Typescript sees queryRequest as 'undefined',
        // which it cannot ever be.
        // We respectfully ignore its foolishness.
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        queryRequest[requestPropertyKey] = this[key];
      }
      return queryRequest;
    }, queryRequest)
  }
}