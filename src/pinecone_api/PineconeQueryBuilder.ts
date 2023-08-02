import { MetadataFilters } from "llamaindex";
import { QueryRequest } from "@pinecone-database/pinecone";
import { SparseValues } from "vectors";
import { PineconeMetadataFilters, PineconeMetadataFilterKey } from "./types";


export type PineconeQueryBuilderOptions = {

  // Number of results to return.
  topK: number;

  // Namespace to search in. If none is provided
  // the default namespace will be used.
  namespace?: string;

  // Used to search by a vector id that's already in Pinecone.
  // Exclusive with `vector`.
  id?: string;

  // Vector to search by. Exclusive with `id`.
  vector?: number[];

  // Sparse vector to search by.
  sparseVector?: SparseValues;

  // Whether or not to include values in the response.
  includeValues?: boolean;

  // Whether or not to include metadata in the response.
  includeMetadata?: boolean;


  // Hyperparameter to control the balance between
  // sparse and dense vectors. Query vector will be multiplied
  // by the alpha while sparse vector will be multiplied by
  // (1 - alpha).
  alpha?: number;

  // LlamaIndex metadata filters. We'll convert them to
  // Pinecone metadata filters.
  filters?: MetadataFilters;
}

// Because we're storing properties directly on the PineconeQueryBuilder,
// we build a type that is the intersection of those on the builder and the
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
  sparseVector?: SparseValues;
  vector?: number[];
  filters?: PineconeMetadataFilters = {};
  alpha?: number;
  id?: string;

  private OPTIONAL_QUERY_REQUEST_PROPERTIES: OptionalRequestProperty =
    ["namespace", "vector", "id", "sparseVector"];

  constructor(options: PineconeQueryBuilderOptions) {
    if (!options.id && !options.vector) {
      throw new Error('One of `id` or `vector` are required.');
    } else if (options.id && options.vector) {
      throw new Error('Only one of `id` and `vector` are allowed.');
    }
    this.topK = options.topK;

    // These are optional values that are okay to be undefined
    this.id = options.id;
    this.vector = options.vector;
    this.namespace = options.namespace;
    this.sparseVector = options.sparseVector;

    // We build the pinecone filters here because we can know immediately
    // if they're invalid. We'll throw before things get out of hand.
    if (options.filters)
      this.filters = this.buildPineconeFilters(options.filters)

    if (options.includeMetadata !== undefined)
      this.includeMetadata = options.includeMetadata!;
    if (options.includeValues !== undefined)
      this.includeValues = options.includeValues!;
    if (options.alpha !== undefined)
      this.alpha = options.alpha!;
  }

  toQueryRequest(): QueryRequest {
    let newQueryRequest: Partial<QueryRequest> = {
      topK: this.topK,
      includeValues: this.includeValues,
      includeMetadata: this.includeMetadata,
    }
    // "Let him cook." — Jose Valim, probably
    newQueryRequest =
      this.addOptionalQueryRequestValues(
        this.buildSparseValues(
          this.processVector(newQueryRequest)));
    return newQueryRequest as QueryRequest;
  }

  buildSparseValues(newQueryRequest: Partial<QueryRequest>): Partial<QueryRequest> {
    if (this.sparseVector && this.alpha) {
      newQueryRequest.sparseVector = {
        indices: this.sparseVector.indices,
        values: this.sparseVector.values.map((value) => value * (1 - this.alpha!))
      }
    }
    return newQueryRequest;
  }

  addOptionalQueryRequestValues(queryRequest: Partial<QueryRequest>): Partial<QueryRequest> {
    if (this.vector) this.processVector(queryRequest);
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

  processVector(queryRequest: Partial<QueryRequest>): Partial<QueryRequest> {
    if (this.vector && this.alpha) {
      queryRequest.vector = queryRequest.vector?.map((value) => value * this.alpha!);
    }
    return queryRequest;
  }

  buildPineconeFilters(metadataFilters: MetadataFilters): PineconeMetadataFilters {
    const pineconeFilter: PineconeMetadataFilters = {};
    if (metadataFilters) {

      // Loop over each filter and convert them to Pinecone filters
      for (const filter of metadataFilters.filters) {
        // switch (filter.filterType) {
        //   case "ExactMatch":
        pineconeFilter[filter.key] = {
          [PineconeMetadataFilterKey.EqualTo]: filter.value
        };
        //       break;
        //     default:
        //       throw new Error(`Filter type ${filter} is not supported.`);
        //   }
      }
    }
    return pineconeFilter
  }
}